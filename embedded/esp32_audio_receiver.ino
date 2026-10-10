#include <WiFi.h>
#include <WebSocketsServer.h>
#include <driver/dac.h>

// --- CONFIGURATION ---
const char* ssid = "YOUR_WIFI_HOTSPOT_NAME";       // ⚠️ Change this
const char* password = "YOUR_WIFI_PASSWORD";         // ⚠️ Change this

WebSocketsServer webSocket = WebSocketsServer(81);

// --- AUDIO BUFFER (Double Buffering to prevent stutter) ---
// We use two buffers: one is being filled by WebSocket, the other is being played.
#define BUFFER_SIZE 4096
uint8_t bufferA[BUFFER_SIZE];
uint8_t bufferB[BUFFER_SIZE];
volatile uint16_t bufferA_len = 0;
volatile uint16_t bufferB_len = 0;
volatile bool useBufferA = true;       // Which buffer is currently being played
volatile bool newDataReady = false;    // Flag: new data has arrived

// ESP32 DAC output on Pin 25
// Wiring: Pin 25 -> Speaker/Amplifier positive, GND -> Speaker/Amplifier negative

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Phone Disconnected!\n", num);
            break;

        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[%u] Phone Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
            }
            break;

        case WStype_BIN:
            // Phone is sending filtered 8-bit PCM audio at 8000Hz
            // Copy into the inactive buffer
            {
                uint8_t* targetBuffer;
                volatile uint16_t* targetLen;

                if (useBufferA) {
                    // Currently playing A, so fill B
                    targetBuffer = bufferB;
                    targetLen = &bufferB_len;
                } else {
                    // Currently playing B, so fill A
                    targetBuffer = bufferA;
                    targetLen = &bufferA_len;
                }

                uint16_t copyLen = (length > BUFFER_SIZE) ? BUFFER_SIZE : length;
                memcpy(targetBuffer, payload, copyLen);
                *targetLen = copyLen;
                newDataReady = true;
            }
            break;

        default:
            break;
    }
}

void setup() {
    Serial.begin(115200);

    // Enable ESP32 internal DAC on Pin 25
    dac_output_enable(DAC_CHANNEL_1);

    Serial.println("Connecting to Wi-Fi...");
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi connected.");
    Serial.print("ESP32 IP Address: ");
    Serial.println(WiFi.localIP());

    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
    Serial.println("WebSocket Audio Server Started on Port 81.");
    Serial.println("Waiting for Phone to connect...");
}

void loop() {
    // Always keep WebSocket alive
    webSocket.loop();

    // If new audio data is ready, swap buffers and play
    if (newDataReady) {
        newDataReady = false;
        useBufferA = !useBufferA; // Swap

        // Play the buffer that was just filled
        uint8_t* playBuffer;
        uint16_t playLen;

        if (useBufferA) {
            playBuffer = bufferA;
            playLen = bufferA_len;
        } else {
            playBuffer = bufferB;
            playLen = bufferB_len;
        }

        // Output each sample to DAC at 8000Hz (125 microseconds per sample)
        for (uint16_t i = 0; i < playLen; i++) {
            dac_output_voltage(DAC_CHANNEL_1, playBuffer[i]);
            delayMicroseconds(125);
        }
    }
}
