/*
 * UNO Q App Lab MCU sketch.
 *
 * Runs on the STM32U585 inside the UNO Q. It reads the trunk IMU, monitors
 * the "I"m OK" fall switch, and accepts cue / LED commands from the Linux side
 * via the Bridge library. The Linux side is `firmware/unoq_app_lab/app.py`.
 */

#include <Bridge.h>
#include <Wire.h>

#define MOTOR2_PIN  5
#define LASER2_PIN  6
#define SW_OK_PIN   2
#define MPU_ADDR    0x68

String bridgeBuffer = "";

void mpuInit() {
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x08); Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); Wire.write(0x10); Wire.endTransmission(true);
}

void readIMU(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  int16_t raw[6];
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  for (int i = 0; i < 7; i++) {
    uint8_t h = Wire.read();
    uint8_t l = Wire.read();
    int16_t v = (h << 8) | l;
    if (i < 6) raw[i] = v;
  }
  ax = raw[0] * (1000.0 / 8192.0);
  ay = raw[1] * (1000.0 / 8192.0);
  az = raw[2] * (1000.0 / 8192.0);
  gx = raw[3] * (1000.0 / 32.8);
  gy = raw[4] * (1000.0 / 32.8);
  gz = raw[5] * (1000.0 / 32.8);
}

void setup() {
  Serial.begin(115200);
  Bridge.begin();
  pinMode(MOTOR2_PIN, OUTPUT); pinMode(LASER2_PIN, OUTPUT);
  pinMode(SW_OK_PIN, INPUT_PULLUP);
  digitalWrite(MOTOR2_PIN, LOW); digitalWrite(LASER2_PIN, LOW);
  mpuInit();
  Serial.println("UNO Q App Lab MCU ready");
}

void loop() {
  // 1. Read and stream IMU at 100 Hz
  static unsigned long lastSample = micros();
  if (micros() - lastSample >= 10000) {
    lastSample += 10000;
    float ax, ay, az, gx, gy, gz;
    readIMU(ax, ay, az, gx, gy, gz);
    Bridge.put(
      "trunk_imu",
      String(ax, 2) + "," + String(ay, 2) + "," + String(az, 2) + ","
      + String(gx, 2) + "," + String(gy, 2) + "," + String(gz, 2)
    );
    // Fall "I'm OK" switch
    if (digitalRead(SW_OK_PIN) == LOW) {
      Bridge.put("fall_ok", "1");
    }
  }

  // 2. Commands from Linux
  Bridge.get("cue_mask", bridgeBuffer, 10);
  if (bridgeBuffer.length() > 0) {
    int mask = bridgeBuffer.toInt();
    digitalWrite(MOTOR2_PIN, (mask & 0x01) ? HIGH : LOW);
    digitalWrite(LASER2_PIN, (mask & 0x02) ? HIGH : LOW);
    bridgeBuffer = "";
  }

  Bridge.get("status_led", bridgeBuffer, 10);
  if (bridgeBuffer.length() > 0) {
    digitalWrite(LED_BUILTIN, bridgeBuffer.toInt() ? HIGH : LOW);
    bridgeBuffer = "";
  }
}
