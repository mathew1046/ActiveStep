#include <WiFi.h>
#include <WiFiUdp.h>

// --- WIFI SETTINGS ---
const char* ssid = "Saju";
const char* password = "sajusaju";

// --- UDP NETWORKING ---
WiFiUDP udp;
const int listenPort = 9999;     // Port to listen for cue commands

// Laptop running `python -m unoq.service` relays cue commands to this node and
// receives a heartbeat so it always knows where Node 1 lives.
IPAddress laptopIP(10, 28, 43, 66);
const int dashboardPort = 5005;
uint32_t hbSeq = 0;
unsigned long lastHeartbeat = 0;

// Hardware Definitions
#define MOTOR_PIN D3

// Vibration duration for commands received from Node 2
#define VIBRATION_DURATION 4000

bool vibrationActive = false;
unsigned long vibrationStart = 0;
bool motorTestActive = false;
unsigned long motorTestStart = 0;

// Packet received from Node 2 (via the laptop relay)
typedef struct struct_command {
  int cmd; // 1 = Trigger, 2 = Stop
} struct_command;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  // Bounded wait — don't hang when powered without a USB serial monitor.
  const unsigned long serialWait = millis();
  while (!Serial && millis() - serialWait < 1500) { delay(10); }

  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  // Connect to Local Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected!");
  Serial.print("Node 1 IP: ");
  Serial.println(WiFi.localIP());

  // Start UDP listener for cue commands
  udp.begin(listenPort);

  Serial.println("Node 1 (Vibration) Ready. Listening for cue commands...");
}

void startVibration() {
  vibrationActive = true;
  vibrationStart = millis();
  digitalWrite(MOTOR_PIN, HIGH);
  Serial.println("[MOTOR] Node 1 vibration ON for 4 seconds.");
}

void stopVibration() {
  vibrationActive = false;
  digitalWrite(MOTOR_PIN, LOW);
}

void checkSerialMotorTest() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.equalsIgnoreCase("motor")) {
      stopVibration();
      motorTestActive = true;
      motorTestStart = millis();
      digitalWrite(MOTOR_PIN, HIGH);
      Serial.println("[TEST] Motor ON for 2 seconds.");
    }
  }

  if (motorTestActive && millis() - motorTestStart >= 2000) {
    motorTestActive = false;
    digitalWrite(MOTOR_PIN, LOW);
    Serial.println("[TEST] Motor OFF.");
  }
}

void updateVibration() {
  if (vibrationActive && millis() - vibrationStart >= VIBRATION_DURATION) {
    stopVibration();
    Serial.println("[MOTOR] Node 1 vibration OFF.");
  }
}

void handleIncomingCommands() {
  int packetSize = udp.parsePacket();
  if (packetSize >= sizeof(struct_command)) {
    struct_command rec;
    udp.read((unsigned char*)&rec, sizeof(rec));

    if (rec.cmd == 1) {
      Serial.println("[UDP] CUE TRIGGER RECEIVED!");
      motorTestActive = false;
      startVibration();
    } else if (rec.cmd == 2) {
      Serial.println("[UDP] VIBRATION CANCELLED.");
      motorTestActive = false;
      stopVibration();
    }
  }
}

// Heartbeat to the laptop: announces this node's IP to the command relay and
// reports current vibration state to the dashboard.
void sendHeartbeat() {
  String j = "{\"seq\":" + String(hbSeq++) +
             ",\"t_ms\":" + String(millis()) +
             ",\"node\":\"node1\"" +
             ",\"vibration\":" + String(vibrationActive ? 1 : 0) +
             ",\"rssi\":" + String(WiFi.RSSI()) + "}";
  udp.beginPacket(laptopIP, dashboardPort);
  udp.print(j);
  udp.endPacket();
}

void loop() {
  checkSerialMotorTest();
  handleIncomingCommands();

  if (millis() - lastHeartbeat >= 1000) {
    lastHeartbeat = millis();
    sendHeartbeat();
  }

  if (!motorTestActive) {
    updateVibration();
  }
  delay(2);
}
