/*
 * ActiveStep - ESP32 Tier-1 FOG detector and cueing unit.
 *
 * Runs on ESP32 DevKit V1 (30 pin).
 * - Samples MPU-6050 shank IMU at 100 Hz
 * - Computes Freeze Index every 250 ms on a 2 s window
 * - Runs the TFLite Micro 1D-CNN (model_quantized.h)
 * - Fires vibration, laser cues with a bandit-selected modality mask
 * - Sends telemetry (IMU batches, pFOG, events, labels) to UNO Q via WiFi UDP
 * - Listens for UNO Q commands (threshold offset, modality, tempo, OTA)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include "model_quantized.h"

// Pins (see plan.md)
#define PIN_MOTOR      25
#define PIN_LASER      26
#define PIN_SW_TRUE    32
#define PIN_SW_FALSE   33
#define PIN_LED        2
#define MPU_ADDR       0x68

// Network
const char* ssid       = "ActiveStep";
const char* password   = "activestep";
const int   localPort  = 5006;  // commands from UNO Q
const int   destPort   = 5005;  // ingest on UNO Q
const char* destIP     = "192.168.4.1"; // UNO Q AP

WiFiUDP udp;

// TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Sampling
const int FS = 100;
const int WIN = FS * 2;                 // 200 samples
const int HOP = FS / 4;                 // 25 samples -> 4 Hz inference
int16_t accRing[WIN][3];                // mg, 16-bit
int ringIndex = 0;
int sampleCount = 0;

// State
enum CueState { IDLE, CUEING, RECOVERING };
CueState cueState = IDLE;
float threshold = 0.7f;
float thresholdOffset = 0.0f;
uint8_t modalityMask = 0x07;            // all by default
float metronomeBPM = 0.0f;
unsigned long cueStartMs = 0;
unsigned long lastTelemMs = 0;

// Per-axis mean/std for input normalization (from model_meta.json -> code)
float mean[3] = { -97.46f, 1009.70f, 242.57f };
float stdv[3] = { 568.88f, 360.12f, 318.51f };

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

void mpuInit() {
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission(true);
  // ±4g, 1000 deg/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x08); Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); Wire.write(0x10); Wire.endTransmission(true);
}

void mpuRead(int16_t* ax, int16_t* ay, int16_t* az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  *ax = (Wire.read() << 8) | Wire.read();
  *ay = (Wire.read() << 8) | Wire.read();
  *az = (Wire.read() << 8) | Wire.read();
  // 16-bit raw with ±2g default? We use ±4g, 1 LSB = 2048 LSB/g ≈ 0.5 mg? Not exact.
  // For the model we rely on scale matching the Daphnet mg scale; adjust with a gain.
}

float fiCompute() {
  // Very small DFT for 3-8 Hz / 0.5-3 Hz power ratio on the 3 axes.
  float totalFreeze[3] = {0,0,0}, totalLoco[3] = {0,0,0};
  // Use a 64-bin DFT at 100 Hz; keep only the bands.
  const int N = WIN;
  const float pi2 = 2.0f * PI;
  for (int axis = 0; axis < 3; axis++) {
    float power[100] = {0};
    for (int f = 0; f < 50; f++) { // 0..49 Hz
      float re = 0, im = 0;
      for (int n = 0; n < N; n++) {
        float val = (accRing[n][axis] - mean[axis]) / (stdv[axis] + 1e-6f);
        float ang = pi2 * f * n / N;
        re += val * cos(ang);
        im -= val * sin(ang);
      }
      power[f] = re*re + im*im;
    }
    for (int f = 0; f < 50; f++) {
      if (f >= 1 && f <= 3) totalLoco[axis] += power[f];
      if (f >= 3 && f <= 8) totalFreeze[axis] += power[f];
    }
  }
  float fi = 0;
  for (int i = 0; i < 3; i++) {
    fi += (totalLoco[i] > 1e-6f) ? totalFreeze[i] / totalLoco[i] : 0;
  }
  return fi / 3.0f;
}

float pCnn() {
  TfLiteTensor* input = interpreter->input(0);
  for (int t = 0; t < WIN; t++) {
    int idx = (ringIndex + t) % WIN;
    for (int a = 0; a < 3; a++) {
      float v = (accRing[idx][a] - mean[a]) / (stdv[a] + 1e-6f);
      input->data.f[t * 3 + a] = v;
    }
  }
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) return 0.0f;
  return interpreter->output(0)->data.f[0];
}

void setupModel() {
  model = tflite::GetModel(g_model_quantized);
  tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
}

void sendTelemetry(float fi, float pfog, const char* state, JsonObject* ev = nullptr) {
  StaticJsonDocument<1024> doc;
  doc["seq"] = millis();
  doc["t_ms"] = millis();
  JsonArray imu = doc.createNestedArray("imu");
  for (int t = WIN - 20; t < WIN; t++) {
    int idx = (ringIndex + t) % WIN;
    JsonArray s = imu.createNestedArray();
    s.add(accRing[idx][0]); s.add(accRing[idx][1]); s.add(accRing[idx][2]);
  }
  doc["fi"] = fi;
  doc["pfog"] = pfog;
  doc["state"] = state;
  if (ev) doc["event"] = *ev;

  char buf[2048];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  udp.beginPacket(destIP, destPort);
  udp.write((uint8_t*)buf, n);
  udp.endPacket();
}

void fireCue() {
  digitalWrite(PIN_LED, HIGH);
  if (modalityMask & 0x01) digitalWrite(PIN_MOTOR, HIGH);
  if (modalityMask & 0x02) digitalWrite(PIN_LASER, HIGH);
}

void stopCue() {
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_MOTOR, LOW);
  digitalWrite(PIN_LASER, LOW);
}

void checkCommands() {
  int n = udp.parsePacket();
  if (!n) return;
  char buf[512]; int r = udp.read(buf, sizeof(buf) - 1); buf[r] = 0;
  StaticJsonDocument<512> doc;
  deserializeJson(doc, buf);
  if (doc.containsKey("threshold_offset")) thresholdOffset = doc["threshold_offset"];
  if (doc.containsKey("modality_mask")) modalityMask = doc["modality_mask"];
  if (doc.containsKey("tempo_bpm")) metronomeBPM = doc["tempo_bpm"];
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_MOTOR, OUTPUT); pinMode(PIN_LASER, OUTPUT); pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_SW_TRUE, INPUT_PULLUP); pinMode(PIN_SW_FALSE, INPUT_PULLUP);
  digitalWrite(PIN_MOTOR, LOW); digitalWrite(PIN_LASER, LOW);

  mpuInit();
  setupModel();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(100); }
  udp.begin(localPort);

  Serial.println("ESP32 ActiveStep ready");
}

void loop() {
  // 1. Sample
  static unsigned long lastSample = micros();
  if (micros() - lastSample >= 10000) { // 100 Hz
    lastSample += 10000;
    int16_t ax, ay, az;
    mpuRead(&ax, &ay, &az);
    // Convert raw to mg. GY-521 default ±2g sensitivity is ~16384 LSB/g.
    // With AFS_SEL=1 (±4g) sensitivity is ~8192 LSB/g.
    const float gain = 1000.0f / 8192.0f;
    accRing[ringIndex][0] = (int16_t)(ax * gain);
    accRing[ringIndex][1] = (int16_t)(ay * gain);
    accRing[ringIndex][2] = (int16_t)(az * gain);
    ringIndex = (ringIndex + 1) % WIN;
    sampleCount++;
  }

  // 2. Inference every HOP samples (~4 Hz)
  if (sampleCount >= HOP) {
    sampleCount = 0;
    float fi = fiCompute();
    float pcnn = pCnn();
    float pfog = 0.6f * pcnn + 0.4f * fminf(fi / 5.0f, 1.0f); // simple blend
    float effectiveThr = threshold + thresholdOffset;

    // Cue FSM
    const char* st = "IDLE";
    unsigned long now = millis();
    if (cueState == IDLE && pfog >= effectiveThr) {
      cueState = CUEING;
      cueStartMs = now;
      fireCue();
      StaticJsonDocument<256> ev;
      ev["type"] = "cue_start";
      ev["modality"] = modalityMask;
      sendTelemetry(fi, pfog, "CUEING", &ev.as<JsonObject>());
      st = "CUEING";
    } else if (cueState == CUEING && pfog < effectiveThr - 0.15f) {
      cueState = RECOVERING;
      unsigned long recovery = now - cueStartMs;
      stopCue();
      StaticJsonDocument<256> ev;
      ev["type"] = "cue_stop";
      ev["recovery_ms"] = recovery;
      sendTelemetry(fi, pfog, "IDLE", &ev.as<JsonObject>());
      st = "IDLE";
      cueState = IDLE;
    } else if (cueState == CUEING) {
      st = "CUEING";
    }

    // 3. Label switches (active 10 s after each cue)
    static unsigned long labelWindowEnd = 0;
    if (st == "CUEING") labelWindowEnd = now + 10000;
    if (now < labelWindowEnd && !digitalRead(PIN_SW_TRUE)) {
      StaticJsonDocument<256> ev;
      ev["type"] = "label"; ev["label"] = "TRUE";
      sendTelemetry(fi, pfog, "IDLE", &ev.as<JsonObject>());
      labelWindowEnd = 0;
    } else if (now < labelWindowEnd && !digitalRead(PIN_SW_FALSE)) {
      StaticJsonDocument<256> ev;
      ev["type"] = "label"; ev["label"] = "FALSE";
      sendTelemetry(fi, pfog, "IDLE", &ev.as<JsonObject>());
      labelWindowEnd = 0;
    }

    // 4. Periodic telemetry
    if (now - lastTelemMs > 200) {
      lastTelemMs = now;
      sendTelemetry(fi, pfog, st);
    }

    // 5. Poll UNO Q commands
    checkCommands();
  }
}
