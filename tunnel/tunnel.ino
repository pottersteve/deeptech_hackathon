/*
  Straight tunnel: drive forward, stay centred between the left and right walls.
    steer = KP * (left - right)   -> closer to one wall = turn away from it
  Side sensors point straight sideways (~90 deg), so readings = distance to each wall.
  Missed echo = keep the last good reading. Wall ahead = stop.

  Logging over Wi-Fi: run  python3 tunnel_mapper.py  on the laptop.
  The robot only drives while the mapper is connected (close it = robot stops).
  No Wi-Fi -> drives anyway after 3 s, logging to USB serial only.
  Log line: S,<millis>,<left_cm>,<front_cm>,<right_cm>,<pwm_left>,<pwm_right>   (-1 = no echo)
*/
#include <WiFiS3.h>
#include "arduino_secrets.h"

// --- Motors: HW-166, left = A, right = B ---
const int PWMA = 5, AIN1 = 8,  AIN2 = 9;
const int PWMB = 6, BIN1 = 10, BIN2 = 11;
const int STBY = 7;
const int MOTOR_DIR = -1;   // motors are wired backwards; set to 1 if you swap the motor wires

// --- Sensors ---
const int L_TRIG = 2,  L_ECHO = 3;
const int F_TRIG = 4,  F_ECHO = 12;
const int R_TRIG = A0, R_ECHO = A1;

// --- Tuning ---
const int   BASE      = 60;    // forward speed (PWM 0-255)
const float KP        = -4;    // steering strength; negative because left/right turned out swapped on the robot
const int   MAX_STEER = 30;    // never steer harder than this
const float FRONT_STOP = 12;   // wall ahead closer than this -> stop

float lastL = 0, lastR = 0;    // last good side readings
int pwmL = 0, pwmR = 0;

WiFiServer server(23);         // log port
WiFiClient logger;
bool wifiOk = false;

void motor(int pwm, int in1, int in2, int speed) {   // -255 to 255
  speed = constrain(speed, -255, 255) * MOTOR_DIR;
  digitalWrite(in1, speed > 0 ? HIGH : LOW);
  digitalWrite(in2, speed < 0 ? HIGH : LOW);
  analogWrite(pwm, abs(speed));
}

void drive(int l, int r) {
  pwmL = l;
  pwmR = r;
  motor(PWMA, AIN1, AIN2, l);
  motor(PWMB, BIN1, BIN2, r);
}

float readCm(int trig, int echo) {   // -1 = no echo
  digitalWrite(trig, LOW);  delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 12000);
  return us == 0 ? -1 : us * 0.0343 / 2;
}

void setup() {
  Serial.begin(115200);
  int pins[] = {PWMA, AIN1, AIN2, PWMB, BIN1, BIN2, STBY, L_TRIG, F_TRIG, R_TRIG};
  for (int p : pins) pinMode(p, OUTPUT);
  pinMode(L_ECHO, INPUT);
  pinMode(F_ECHO, INPUT);
  pinMode(R_ECHO, INPUT);
  digitalWrite(STBY, HIGH);
  drive(0, 0);

  Serial.print("Connecting to "); Serial.println(SECRET_SSID);
  WiFi.begin(SECRET_SSID, SECRET_PASS);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) delay(500);
  wifiOk = WiFi.status() == WL_CONNECTED;
  if (wifiOk) {
    while (WiFi.localIP() == IPAddress(0, 0, 0, 0)) delay(100);
    server.begin();
    Serial.print("Log server: "); Serial.print(WiFi.localIP()); Serial.println(":23 - waiting for tunnel_mapper.py");
  } else {
    Serial.println("No Wi-Fi - starting in 3 s, serial log only");
    delay(3000);
  }
}

void loop() {
  if (wifiOk && !logger.connected()) {
    drive(0, 0);                      // nobody watching -> don't drive
    logger = server.accept();
    if (logger.connected()) logger.println("# tunnel robot connected");
    delay(50);
    return;
  }

  float left  = readCm(L_TRIG, L_ECHO);  delay(10);
  float front = readCm(F_TRIG, F_ECHO);  delay(10);
  float right = readCm(R_TRIG, R_ECHO);

  if (left  > 0) lastL = left;    // missed echo -> trust the last one
  if (right > 0) lastR = right;

  if (front > 0 && front < FRONT_STOP) {
    drive(0, 0);
  } else {
    int steer = constrain(KP * (lastL - lastR), -MAX_STEER, MAX_STEER);
    drive(BASE - steer, BASE + steer);   // left wall farther -> steer left
  }

  String line = "S," + String(millis()) + "," + String(left, 1) + "," + String(front, 1) + "," +
                String(right, 1) + "," + String(pwmL) + "," + String(pwmR);
  Serial.println(line);
  if (logger.connected()) logger.println(line);
  delay(20);
}
