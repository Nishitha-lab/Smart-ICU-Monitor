#include <Wire.h>
#include <HX711.h>
#include "MAX30100_PulseOximeter.h"

// =====================================================
// PIN CONNECTIONS
// =====================================================

#define HX711_DT    4
#define HX711_SCK   5

#define I2C_SDA     21
#define I2C_SCL     22

#define BUZZER_PIN  18


// =====================================================
// HX711 LOAD CELL
// =====================================================

HX711 scale;

float CALIBRATION_FACTOR = 117.72;
long TARE_OFFSET = 168618;


// =====================================================
// MAX30100 HEART RATE SENSOR
// =====================================================

PulseOximeter pox;

const unsigned long REPORTING_PERIOD_MS = 1000;

// Time of the most recent detected heartbeat
unsigned long lastBeatDetectedAt = 0;

// Has the sensor detected at least one heartbeat?
bool heartbeatDetectedOnce = false;

// Allow 5 seconds without a detected beat before showing 0
const unsigned long HEARTBEAT_TIMEOUT_MS = 5000;


// =====================================================
// WEIGHT SMOOTHING
// =====================================================

float filteredWeight = 0.0;

const float SMOOTHING_FACTOR = 0.20;


// =====================================================
// FLUID THRESHOLDS
// =====================================================

const float WARNING_LEVEL = 500.0;
const float CRITICAL_LEVEL = 200.0;


// =====================================================
// ANOMALY DETECTION
// =====================================================

const float ANOMALY_PERCENT = 25.0;

float previousWeight = -1.0;
float anomalyPreviousWeight = 0.0;

bool anomalyDetected = false;


// =====================================================
// SYSTEM STATUS
// =====================================================

String systemStatus = "NORMAL";


// =====================================================
// BUZZER
// =====================================================

bool buzzerState = false;

unsigned long lastBuzzerChange = 0;

const unsigned long BEEP_ON_TIME = 200;


// =====================================================
// HEARTBEAT CALLBACK
// =====================================================

void onBeatDetected()
{
  lastBeatDetectedAt = millis();

  heartbeatDetectedOnce = true;

  Serial.println("BEAT!");
}


// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("       IV MONITORING SYSTEM");
  Serial.println("========================================");
  Serial.println(" HX711 + MAX30100 + BUZZER");
  Serial.println("========================================");


  // Initialize buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("Buzzer initialized");


  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);

  delay(100);

  Serial.println("I2C initialized");


  // Initialize HX711
  Serial.println("Starting HX711...");

  scale.begin(HX711_DT, HX711_SCK);

  delay(500);

  if (scale.is_ready())
  {
    Serial.println("HX711 connected!");
  }
  else
  {
    Serial.println("HX711 NOT FOUND!");
  }


  // Initialize MAX30100
  Serial.println("Starting MAX30100...");

  if (!pox.begin())
  {
    Serial.println("MAX30100 FAILED!");
  }
  else
  {
    Serial.println("MAX30100 connected!");

    pox.setOnBeatDetectedCallback(onBeatDetected);
  }


  Serial.println();
  Serial.println("----------------------------------------");
  Serial.println("Place IV bottle on load cell");
  Serial.println("Place ONE finger on MAX30100");
  Serial.println("----------------------------------------");
  Serial.println();
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  // Keep this running continuously for MAX30100
  pox.update();


  // ===================================================
  // READ LOAD CELL EVERY 300 MILLISECONDS
  // ===================================================

  static unsigned long lastWeightRead = 0;

  if (millis() - lastWeightRead >= 300)
  {
    lastWeightRead = millis();

    if (scale.is_ready())
    {
      long rawValue = scale.read();

      float weightGrams =
        (rawValue - TARE_OFFSET) / CALIBRATION_FACTOR;

      if (weightGrams < 0)
      {
        weightGrams = 0;
      }

      // Smooth the readings
      if (filteredWeight == 0)
      {
        filteredWeight = weightGrams;
      }
      else
      {
        filteredWeight =
          (SMOOTHING_FACTOR * weightGrams) +
          ((1.0 - SMOOTHING_FACTOR) * filteredWeight);
      }
    }
  }


  // ===================================================
  // ANOMALY DETECTION
  // ===================================================

  anomalyDetected = false;

  float currentWeight = filteredWeight;

  if (previousWeight > 0 && currentWeight > 0)
  {
    float decrease = previousWeight - currentWeight;

    float decreasePercent =
      (decrease / previousWeight) * 100.0;

    if (decrease > 0 &&
        decreasePercent >= ANOMALY_PERCENT)
    {
      anomalyDetected = true;

      anomalyPreviousWeight = previousWeight;
    }
  }

  if (currentWeight > 0)
  {
    previousWeight = currentWeight;
  }


  // ===================================================
  // DETERMINE FLUID STATUS
  // ===================================================

  if (anomalyDetected)
  {
    systemStatus = "ABNORMAL";
  }
  else if (currentWeight <= CRITICAL_LEVEL)
  {
    systemStatus = "CRITICAL";
  }
  else if (currentWeight <= WARNING_LEVEL)
  {
    systemStatus = "WARNING";
  }
  else
  {
    systemStatus = "NORMAL";
  }


  // Update buzzer without blocking the sensor
  updateBuzzer();


  // ===================================================
  // REPORT SENSOR VALUES EVERY SECOND
  // ===================================================

  static unsigned long lastReport = 0;

  if (millis() - lastReport >= REPORTING_PERIOD_MS)
  {
    lastReport = millis();


    // Get current sensor readings
    float bpm = pox.getHeartRate();

    float spo2 = pox.getSpO2();


    // ================================================
    // HEART RATE FIX
    // ================================================

    // Only apply the timeout after the first detected beat.
    // This avoids forcing BPM to zero before the first beat.

    if (heartbeatDetectedOnce &&
        millis() - lastBeatDetectedAt > HEARTBEAT_TIMEOUT_MS)
    {
      bpm = 0;
    }


    // Reject invalid heart-rate values
    if (bpm < 40 || bpm > 200)
    {
      bpm = 0;
    }


    // Reject invalid SpO2 values
    if (spo2 < 70 || spo2 > 100)
    {
      spo2 = 0;
    }


    // ================================================
    // SERIAL OUTPUT FOR FLASK
    // ================================================

    Serial.print("Fluid: ");
    Serial.print(currentWeight, 1);

    Serial.print(" mL | BPM: ");
    Serial.print(bpm, 1);

    Serial.print(" | SpO2: ");
    Serial.print(spo2, 1);

    Serial.print("% | STATUS: ");
    Serial.println(systemStatus);


    // Print anomaly details when necessary
    if (anomalyDetected)
    {
      Serial.print("ANOMALY: Previous=");

      Serial.print(anomalyPreviousWeight, 1);

      Serial.print(" mL | Current=");

      Serial.print(currentWeight, 1);

      Serial.println(" mL");
    }
  }
}


// =====================================================
// BUZZER CONTROL
// =====================================================

void updateBuzzer()
{
  // NORMAL: silent
  if (systemStatus == "NORMAL")
  {
    digitalWrite(BUZZER_PIN, LOW);

    buzzerState = false;

    return;
  }


  // WARNING: silent
  if (systemStatus == "WARNING")
  {
    digitalWrite(BUZZER_PIN, LOW);

    buzzerState = false;

    return;
  }


  unsigned long now = millis();


  // CRITICAL: intermittent beep
  if (systemStatus == "CRITICAL")
  {
    if (!buzzerState)
    {
      if (now - lastBuzzerChange >= 1000)
      {
        digitalWrite(BUZZER_PIN, HIGH);

        buzzerState = true;

        lastBuzzerChange = now;
      }
    }
    else
    {
      if (now - lastBuzzerChange >= BEEP_ON_TIME)
      {
        digitalWrite(BUZZER_PIN, LOW);

        buzzerState = false;

        lastBuzzerChange = now;
      }
    }
  }


  // ABNORMAL: faster intermittent beep
  else if (systemStatus == "ABNORMAL")
  {
    if (!buzzerState)
    {
      if (now - lastBuzzerChange >= 300)
      {
        digitalWrite(BUZZER_PIN, HIGH);

        buzzerState = true;

        lastBuzzerChange = now;
      }
    }
    else
    {
      if (now - lastBuzzerChange >= BEEP_ON_TIME)
      {
        digitalWrite(BUZZER_PIN, LOW);

        buzzerState = false;

        lastBuzzerChange = now;
      }
    }
  }
}