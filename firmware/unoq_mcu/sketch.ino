/*
 * UNO Q STM32U585 MCU sketch (Zephyr / Arduino Core).
 *
 * - Reads MPU-6050 on the Qwiic/I2C bus at 100 Hz
 * - Runs fall-detection primitives (free-fall -> impact -> stillness)
 * - Drives cue set 2 (motor + laser) and the "I'm OK" switch
 * - Exposes data to the Linux side via Arduino Bridge RPC
 */

#include <Wire.h>

// Arduino UNO Q MCU-side digital pins (D2..D13 as usual)
#define PIN_MOTOR2    5
#define PIN_LASER2    6
#define PIN_SW_OK     2
#define PIN_LED       LED_BUILTIN

#define MPU_ADDR      0x68

// Fall FSM
enum FallState { NONE, SUSPECT, WAIT_OK };
FallState fallState = NONE;
unsigned long suspectStart = 0;

const float G_MS2 = 9.80665f;
float ax, ay, az;              // m/s^2
unsigned long lastSample = 0;

void mpuInit() {
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x08); Wire.endTransmission(true); // ±4g
}

void readIMU() {
  int16_t raw[3];
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  raw[0] = (Wire.read() << 8) | Wire.read();
  raw[1] = (Wire.read() << 8) | Wire.read();
  raw[2] = (Wire.read() << 8) | Wire.read();
  const float gain = G_MS2 / 8192.0f; // ±4g
  ax = raw[0] * gain;
  ay = raw[1] * gain;
  az = raw[2] * gain;
}

float gMag() { return sqrtf(ax*ax + ay*ay + az*az); }

void fireCue2() {
  digitalWrite(PIN_MOTOR2, HIGH);
  digitalWrite(PIN_LASER2, HIGH);
}

void stopCue2() {
  digitalWrite(PIN_MOTOR2, LOW);
  digitalWrite(PIN_LASER2, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_MOTOR2, OUTPUT); pinMode(PIN_LASER2, OUTPUT);
  pinMode(PIN_SW_OK, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_MOTOR2, LOW); digitalWrite(PIN_LASER2, LOW);
  mpuInit();
  Serial.println("UNO Q MCU ready");
}

void loop() {
  // 100 Hz sample
  if (micros() - lastSample >= 10000) {
    lastSample += 10000;
    readIMU();

    float g = gMag();

    // Fall detection primitives
    switch (fallState) {
      case NONE:
        if (g < 0.5f * G_MS2) {
          fallState = SUSPECT;
          suspectStart = millis();
          Serial.println("FALL_SUSPECT");
        }
        break;
      case SUSPECT:
        if (g > 2.5f * G_MS2) {
          // impact
          fallState = WAIT_OK;
          suspectStart = millis();
          Serial.println("FALL_IMPACT");
          digitalWrite(PIN_LED, HIGH);
        } else if (millis() - suspectStart > 500) {
          fallState = NONE; // free-fall too long without impact -> not a fall
        }
        break;
      case WAIT_OK:
        if (millis() - suspectStart > 2000) {
          // 2 s of stillness after impact -> escalate
          if (g > 0.8f * G_MS2 && g < 1.3f * G_MS2) {
            Serial.println("FALL_ALARM");
            // The Linux side will play the check-in prompt and escalate.
          }
          fallState = NONE;
          digitalWrite(PIN_LED, LOW);
        } else if (!digitalRead(PIN_SW_OK)) {
          Serial.println("FALL_OK");
          fallState = NONE;
          digitalWrite(PIN_LED, LOW);
        }
        break;
    }

    // Stream IMU to Linux via Bridge RPC once per 100 ms
    static unsigned long lastBridge = 0;
    if (millis() - lastBridge >= 100) {
      lastBridge = millis();
      // Format: "IMU t ax ay az gx gy gz" (gyro set to 0 for MPU-6050 raw if not read)
      Serial.print("IMU "); Serial.print(lastBridge);
      Serial.print(" "); Serial.print(ax, 2);
      Serial.print(" "); Serial.print(ay, 2);
      Serial.print(" "); Serial.print(az, 2);
      Serial.print(" 0 0 0"); // placeholder gyro
      Serial.println();
    }

    // Execute Linux cue commands on set 2 (future: read from Bridge)
    while (Serial.available()) {
      char c = Serial.read();
      if (c == 'C') fireCue2();
      if (c == 'S') stopCue2();
    }
  }
}
