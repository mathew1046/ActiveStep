# Public dashboard telemetry bridge

The public dashboard accepts authenticated HTTPS telemetry at:

```text
POST https://<dashboard-host>/api/telemetry
X-ActiveStep-Token: <token>
Content-Type: application/json
```

Payload fields accepted from either sensor:

```json
{
  "sensor": "shank",
  "accMag": 1.23,
  "gyroMag": 0.44,
  "fi": 0.80,
  "pfog": 0.62,
  "state": "IDLE",
  "seq": 123
}
```

Use `sensor: "waist"` for the second unit. The existing ESP sketch broadcasts UDP locally; it must additionally forward a small HTTPS JSON packet to this endpoint for the public dashboard. Keep the local UDP path for the offline-first UNO Q operation.

Arduino sketch shape:

```cpp
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

const char* dashboardUrl = "https://<dashboard-host>/api/telemetry";
const char* dashboardToken = "<token>";
unsigned long lastCloudPost = 0;

void postTelemetry(float accMag, float gyroMag, float fi, float pfog,
                   const char* sensor, const char* state) {
  if (millis() - lastCloudPost < 200 || WiFi.status() != WL_CONNECTED) return;
  lastCloudPost = millis();
  WiFiClientSecure client;
  client.setInsecure(); // prototype only; use the server CA for production
  HTTPClient http;
  if (!http.begin(client, dashboardUrl)) return;
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-ActiveStep-Token", dashboardToken);
  String body = "{\"sensor\":\"" + String(sensor) +
    "\",\"accMag\":" + String(accMag, 4) +
    ",\"gyroMag\":" + String(gyroMag, 4) +
    ",\"fi\":" + String(fi, 4) +
    ",\"pfog\":" + String(pfog, 4) +
    ",\"state\":\"" + String(state) + "\"}";
  http.POST(body);
  http.end();
}
```

Call `postTelemetry(...)` at a low rate (5 Hz is enough for the dashboard); do not replace the local 20 Hz UDP/control loop.
