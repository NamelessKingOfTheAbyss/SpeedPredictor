#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
uint16_t BNO055_SAMPLERATE_DELAY_MS = 10;

#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcdef1"

class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!bno.begin()) {
    Serial.println("No BNO055 detected. Check wiring!");
    while (1);
  }

  BLEDevice::init("BNO055_BLE_Sensor");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();
}

void loop() {
  if (deviceConnected) {
    sensors_event_t angVelocityData, linearAccelData;
    bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
    bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);

    char data[100];
    snprintf(data, sizeof(data), "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
             linearAccelData.acceleration.x,
             linearAccelData.acceleration.y,
             linearAccelData.acceleration.z,
             angVelocityData.gyro.x,
             angVelocityData.gyro.y,
             angVelocityData.gyro.z);

    pCharacteristic->setValue((uint8_t*)data, strlen(data));
    pCharacteristic->notify();
    
    delay(BNO055_SAMPLERATE_DELAY_MS);
  } else {
    delay(100);
  }
}
