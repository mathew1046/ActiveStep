#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <TinyGPS++.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <math.h>

// --- WIFI & ALERT SETTINGS ---
const char* ssid = "Saju";
const char* password = "sajusaju";
String alertURL = "http://maker.ifttt.com/trigger/patient_emergency/with/key/YOUR_IFTTT_KEY";

// --- UDP NETWORKING ---
WiFiUDP udp;
const int relayCmdPort = 9999;  // laptop relay forwards commands to Node 1

// Laptop running `python -m unoq.service`: receives telemetry, relays cue
// commands to Node 1, hosts the dashboard.
IPAddress laptopIP(10, 28, 43, 66);
const int dashboardPort = 5005;
uint32_t dashSeq = 0;
unsigned long lastDashTx = 0;

// --- HARDWARE PINS ---
#define BUTTON_PIN 4
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17
// ADXL345 on default I2C (SDA 21, SCL 22)

TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
Adafruit_ADXL345_Unified adxl = Adafruit_ADXL345_Unified(12345);

WebServer server(80);

// --- FOG PREDICTION PARAMETERS ---
float legMotion = 0.0;
float accMag = 0.0;  // gravity-compensated acceleration magnitude

// Latest accelerometer sample + batch uplinked to the dashboard (~100 Hz)
float ax = 0, ay = 0, az = 0;
#define IMU_BATCH_MAX 16
float imuBatch[IMU_BATCH_MAX][6];
int imuCount = 0;
unsigned long lastSample = 0;

// Thresholds
float LEG_WALK_THRESHOLD = 0.40;
#define FREEZE_CONFIRM_TIME 1000
#define CUE_DURATION_MS 4000      // mirror of Node 1 VIBRATION_DURATION

enum GaitState { STATE_IDLE, STATE_WALKING };
GaitState gaitState = STATE_IDLE;

bool possibleFreeze = false;
unsigned long freezeTimer = 0;
bool fogAlertPending = false;
unsigned long lastButtonPress = 0;
unsigned long lastSequencePress = 0;
uint8_t buttonPressCount = 0;
bool lastButtonState = HIGH;

// Cue bookkeeping (mirrors Node 1's vibration window for the dashboard)
unsigned long cueActiveUntil = 0;
unsigned long cueStartMs = 0;

#define BUTTON_DEBOUNCE_MS 40
#define MULTIPRESS_WINDOW_MS 1000

void sendVibrationCommand(int command);
void cueStart(const char* cause);
void cueStop(const char* cause);
void dashboardSend(String extra = "");
void printCurrentLocation();

void setup() {
  Serial.begin(115200);

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Connect to Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.disconnect(true);
  delay(300);
  WiFi.begin(ssid, password);

  const unsigned long wifiStart = millis();
  wl_status_t lastStatus = WL_NO_SHIELD;
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000) {
    wl_status_t status = WiFi.status();
    if (status != lastStatus) {
      Serial.printf("\nWi-Fi status: %d", status);
      lastStatus = status;
    }
    Serial.print(".");
    delay(500);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi Connected!");
    Serial.print("Gateway IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.printf("\nWi-Fi connection timed out; final status: %d\n", WiFi.status());
    Serial.println("Please check SSID/password and ensure 2.4GHz network.");
  }

  // Initialize accelerometer (ADXL345, I2C 0x53)
  Wire.begin();
  if (!adxl.begin()) {
    Serial.println("ERROR: ADXL345 not detected! Check I2C wiring.");
    while (1) { delay(1000); }
  }
  adxl.setRange(ADXL345_RANGE_4_G);
  adxl.setDataRate(ADXL345_DATARATE_100_HZ);
  Serial.println("ADXL345 ready.");

  // HTTP WebServer Endpoints for debugging and control
  server.on("/data", HTTP_GET, []() {
    String json = "{\"legMotion\":" + String(legMotion, 4) +
                  ",\"accMag\":" + String(accMag, 4) +
                  ",\"gaitState\":" + String((int)gaitState) +
                  ",\"cueActive\":" + String(millis() < cueActiveUntil ? 1 : 0) +
                  ",\"wifi\":\"" + WiFi.localIP().toString() + "\"}";
    server.send(200, "application/json", json);
  });

  server.on("/vibrate", HTTP_GET, []() {
    cueStart("remote");
    server.send(200, "text/plain", "Node 1 vibration command sent");
  });

  server.on("/stop", HTTP_GET, []() {
    cueStop("remote");
    server.send(200, "text/plain", "Node 1 vibration stop command sent");
  });

  server.begin();
  Serial.println("HTTP server ready: /data, /vibrate, /stop");
  Serial.println("Node 2 Gateway Engine Ready.");
}

// ---------------------------------------------------------------------------
// Dashboard uplink helpers
// ---------------------------------------------------------------------------

// pFOG shown on the dashboard: ramps 0 -> 1 across the freeze-confirm window,
// held at 1 while a cue is active.
float dashPfog() {
  if (millis() < cueActiveUntil) return 1.0f;
  if (possibleFreeze) {
    return min(1.0f, (millis() - freezeTimer) / (float)FREEZE_CONFIRM_TIME);
  }
  return 0.0f;
}

const char* dashState() {
  if (millis() < cueActiveUntil) return "CUEING";
  return gaitState == STATE_WALKING ? "WALKING" : "IDLE";
}

// Send a JSON telemetry packet to the laptop ingest (:5005).
// `extra` is an optional raw JSON fragment appended before the closing brace,
// e.g. ",\"event\":{\"type\":\"cue_start\",...}".
void dashboardSend(String extra) {
  String j = "{\"seq\":" + String(dashSeq++) +
             ",\"t_ms\":" + String(millis()) +
             ",\"fi\":" + String(legMotion, 4) +
             ",\"pfog\":" + String(dashPfog(), 3) +
             ",\"state\":\"" + String(dashState()) + "\"" +
             ",\"acc_mag\":" + String(accMag, 4) +
             ",\"imu\":[";
  for (int i = 0; i < imuCount; i++) {
    if (i) j += ",";
    j += "[" + String(imuBatch[i][0], 3) + "," + String(imuBatch[i][1], 3) + "," +
         String(imuBatch[i][2], 3) + ",0,0,0]";
  }
  j += "]";
  if (gps.location.isValid()) {
    j += ",\"gps\":[" + String(gps.location.lat(), 6) + "," +
         String(gps.location.lng(), 6) + "]";
  }
  j += extra + "}";
  udp.beginPacket(laptopIP, dashboardPort);
  udp.print(j);
  udp.endPacket();
  imuCount = 0;
}

// Send a cue command to Node 1 through the laptop relay (:9999).
void sendVibrationCommand(int command) {
  struct { int cmd; } cueCmd;
  cueCmd.cmd = command;
  udp.beginPacket(laptopIP, relayCmdPort);
  udp.write((uint8_t *)&cueCmd, sizeof(cueCmd));
  udp.endPacket();
  Serial.printf("[UDP] Sent command %d via relay\n", command);
}

// Start a vibration cue on Node 1 and report cue_start to the dashboard.
// cause: "emergency" | "manual" | "remote"
void cueStart(const char* cause) {
  sendVibrationCommand(1);
  cueStartMs = millis();
  cueActiveUntil = cueStartMs + CUE_DURATION_MS;
  dashboardSend(",\"event\":{\"type\":\"cue_start\",\"modality\":1,\"cause\":\"" +
                String(cause) + "\"}");
}

// Stop the cue on Node 1 and report cue_stop (+recovery_ms) to the dashboard.
void cueStop(const char* cause) {
  sendVibrationCommand(2);
  unsigned long recovery = millis() - cueStartMs;
  cueActiveUntil = 0;
  dashboardSend(",\"recovery_ms\":" + String(recovery) +
                ",\"event\":{\"type\":\"cue_stop\",\"modality\":1,\"cause\":\"" +
                String(cause) + "\"}");
}

// ---------------------------------------------------------------------------

void readAccelerometer() {
  sensors_event_t event;
  adxl.getEvent(&event);
  ax = event.acceleration.x;
  ay = event.acceleration.y;
  az = event.acceleration.z;

  // Gravity-compensated magnitude: deviation from ~1 g.
  float mag = sqrt(ax * ax + ay * ay + az * az);
  accMag = fabs(mag - 9.81f);
  legMotion = accMag;
}

void runFOGDetection() {
  // 1. Is the leg currently moving above the threshold?
  if (legMotion > LEG_WALK_THRESHOLD) {
    gaitState = STATE_WALKING;
    possibleFreeze = false;
  }
  // 2. Motion has dropped below threshold. Were they walking previously?
  else {
    if (gaitState == STATE_WALKING) {
      if (!possibleFreeze) {
        possibleFreeze = true;
        freezeTimer = millis();
        Serial.println("[DETECTION] Sudden leg stop detected. Monitoring for FOG...");
      }

      // If low motion continues for FREEZE_CONFIRM_TIME duration
      if (millis() - freezeTimer >= FREEZE_CONFIRM_TIME) {
        Serial.println("\n*********************************************");
        Serial.println(" [ALARM] FREEZING OF GAIT DETECTED!");
        Serial.println("*********************************************\n");

        // Do not trigger automatic vibration while a button sequence is being
        // entered; a triple press must never be mistaken for a double press.
        if (buttonPressCount == 0) {
          cueStart("emergency"); // Trigger motor on Node 1 + dashboard event
        }

        fogAlertPending = true;
        gaitState = STATE_IDLE; // Reset state
        possibleFreeze = false;
      }
    }
  }
}

void sendAlert(String eventType) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String finalURL = alertURL + "?value1=" + eventType;

    Serial.println("Sending Cloud Alert: " + finalURL);
    http.begin(finalURL);
    int code = http.GET();
    Serial.print("HTTP Result: "); Serial.println(code);
    http.end();
  }
}

void printCurrentLocation() {
  if (gps.location.isValid()) {
    Serial.print("[GPS] Latitude: ");
    Serial.println(gps.location.lat(), 6);
    Serial.print("[GPS] Longitude: ");
    Serial.println(gps.location.lng(), 6);
    Serial.printf("[GPS] Satellites: %lu\n", (unsigned long)gps.satellites.value());
  } else {
    Serial.println("[GPS] Location not available yet. Move outdoors and wait for a fix.");
    Serial.printf("[GPS] Characters processed: %lu, satellites: %lu\n",
                  (unsigned long)gps.charsProcessed(),
                  (unsigned long)gps.satellites.value());
  }
}

void handleButtonPresses() {
  bool buttonState = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  if (lastButtonState == HIGH && buttonState == LOW &&
      now - lastButtonPress >= BUTTON_DEBOUNCE_MS) {
    lastButtonPress = now;

    if (buttonPressCount == 0 || now - lastSequencePress > MULTIPRESS_WINDOW_MS) {
      buttonPressCount = 1;
      if (millis() < cueActiveUntil) {
        // Cancelling a live cue = patient says it was a false alarm.
        cueStop("cancelled");
        dashboardSend(",\"event\":{\"type\":\"label\",\"label\":\"FALSE_POSITIVE\"}");
      } else {
        sendVibrationCommand(2);
      }
      fogAlertPending = false;
      Serial.println("[BUTTON] Single press: vibration cancelled (false positive if cue was live).");
    } else {
      buttonPressCount++;
      Serial.printf("[BUTTON] Press %u detected.\n", buttonPressCount);
    }
    lastSequencePress = now;
  }
  lastButtonState = buttonState;

  // Wait one full window after the LAST press before classifying the sequence.
  if (buttonPressCount > 0 && now - lastSequencePress > MULTIPRESS_WINDOW_MS) {
    if (buttonPressCount == 2) {
      Serial.println("[BUTTON] Double press: manual cue = missed detection (false negative).");
      cueStart("manual");
      dashboardSend(",\"event\":{\"type\":\"label\",\"label\":\"FALSE_NEGATIVE\"}");
    } else if (buttonPressCount >= 3) {
      Serial.println("[BUTTON] Triple press: current GPS location:");
      printCurrentLocation();
    }
    buttonPressCount = 0;
  }
}

void loop() {
  // Handle HTTP requests (/data, /vibrate, /stop)
  server.handleClient();

  // Read accelerometer every loop pass (~10 ms tick)
  readAccelerometer();
  if (millis() - lastSample >= 10 && imuCount < IMU_BATCH_MAX) {
    lastSample = millis();
    imuBatch[imuCount][0] = ax;
    imuBatch[imuCount][1] = ay;
    imuBatch[imuCount][2] = az;
    imuCount++;
  }

  // Read GPS stream
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Run single-sensor decision engine
  runFOGDetection();

  // Single = cancel vibration, double = 4-second manual cue,
  // triple = print GPS location.
  handleButtonPresses();

  // Node 1's 4 s vibration window expired without a cancel -> report cue_stop.
  if (cueActiveUntil && millis() >= cueActiveUntil) {
    unsigned long recovery = millis() - cueStartMs;
    cueActiveUntil = 0;
    dashboardSend(",\"recovery_ms\":" + String(recovery) +
                  ",\"event\":{\"type\":\"cue_stop\",\"modality\":1,\"cause\":\"auto_end\"}");
  }

  // Stream telemetry to the dashboard at 10 Hz.
  if (millis() - lastDashTx >= 100) {
    lastDashTx = millis();
    dashboardSend();
  }

  // Handle Cloud Dispatch
  if (fogAlertPending) {
    sendAlert("FOG_PREDICTED");
    fogAlertPending = false;
  }
  delay(10); // Small delay to prevent watchdog timer resets
}
