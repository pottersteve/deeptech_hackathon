/*
  Tunnel: drive straight ahead, no centring - the tunnel's width can change, so it doesn't try
  to hold the middle. Every ping checks all three sonars:
    side wall closer than SIDE_NEAR -> steer away from it (the closer one if both are)
    wall ahead -> back up a bit, then spin towards the open side (left first); dead end = turn around.
  Side sensors point straight sideways (~90 deg). Pressed against a wall a sonar gets no echo,
  so a side that was near and then misses an echo counts as still near for a few pings.

  It never starts by itself: it waits for "GO" (Space in maze_mapper.py, or type GO in the
  Serial Monitor). "STOP" stops it. Over Wi-Fi it pauses while the mapper is disconnected
  and carries on by itself when it reconnects (unless STOP was sent).
  Log line: S,<millis>,<left_cm>,<front_cm>,<right_cm>,<pwm_left>,<pwm_right>   (-1 = no echo)
  Turn line: T,<millis>,<way>,<ms>   (way 1 = spun left, -1 = right, for <ms> ms)
*/
#include <WiFiS3.h>
#include "arduino_secrets.example.h"

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
const int   SPIN_DIR  = -1;    // which wheel goes forward to spin left; -1 because left/right turned out swapped on the robot
const float FRONT_TURN = 15;   // wall ahead closer than this -> back up and turn (mapping.py: front_turn_cm)
const float OPEN_CM   = 25;    // a side further than this (or no echo) is open   (mapping.py: open_cm)
const int   BACKUP_MS = 300;   // how long to reverse before turning
const int   SPIN      = 80;   // spin speed (PWM)
const int   TURN_90_MS = 200;  // spin this long for 90 deg - calibrate! (mapping.py: spin_deg_s = 90000 / this)
const int   SENSOR_SIDE = 1;   // -1 if the L sensor really faces the robot's right (then run the mapper with --swap-sides)
const float SIDE_NEAR = 8;     // side wall closer than this -> steer away from it
const int   AVOID     = 25;    // how hard to steer away (PWM difference added to / taken from BASE)
const int   STICKY    = 3;     // a near side that misses this many echoes in a row counts as open again

int pwmL = 0, pwmR = 0;
float roomL = OPEN_CM + 1, roomR = OPEN_CM + 1;   // room on each side, cm (see room())
int missL = 0, missR = 0;

WiFiServer server(23);         // log port
WiFiClient logger;
bool wifiOk = false;
bool running = false;          // set by GO, cleared by STOP
String cmd = "";

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

void logLine(String line) {
  Serial.println(line);
  if (logger.connected()) logger.println(line);
}

void logSample(float left, float front, float right) {
  logLine("S," + String(millis()) + "," + String(left, 1) + "," + String(front, 1) + "," +
      String(right, 1) + "," + String(pwmL) + "," + String(pwmR));
}

void handleCommand(String c) {
  c.trim();
  c.toUpperCase();
  if (c == "GO") {
    running = true;
    logLine("# GO");
  } else if (c == "STOP") {
    running = false;
    drive(0, 0);
    logLine("# STOP");
  }
}

void readCommands(Stream &in) {
  while (in.available()) {
    char ch = in.read();
    if (ch == '\n' || ch == '\r') {
      if (cmd.length()) handleCommand(cmd);
      cmd = "";
    } else if (cmd.length() < 32) {
      cmd += ch;
    }
  }
}

// Room on one side from a new reading. A wall reading (up to OPEN_CM) is taken as is. No echo or
// a far one keeps the last value for STICKY pings - right next to a wall the sonar misses or
// reads junk like 190 cm - then the side counts as open.
float room(float d, float &last, int &miss) {
  if (d > 0 && d <= OPEN_CM) { last = d; miss = 0; }
  else if (++miss > STICKY)  last = OPEN_CM + 1;
  return last;
}

void forgetSides() {                 // after a spin the old side readings belong to the old corridor
  roomL = roomR = OPEN_CM + 1;
  missL = missR = 0;
}

void steerTowards(int side) {        // 1 = towards the L sensor's side, -1 = the R sensor's (same sense as spin)
  drive(BASE - SPIN_DIR * AVOID * side, BASE + SPIN_DIR * AVOID * side);
}

void spin(int side) {                // 1 = towards the L sensor's side, -1 = the R sensor's; logged as a T line
  drive(-SPIN_DIR * SPIN * side, SPIN_DIR * SPIN * side);
  delay(TURN_90_MS);
  drive(0, 0);
  logLine("T," + String(millis()) + "," + String(side * SENSOR_SIDE) + "," + String(TURN_90_MS));
  delay(100);                        // let it settle before the next ping
}

void backUpAndTurn() {
  unsigned long until = millis() + BACKUP_MS;   // reverse, still logging so the mapper sees it
  while (millis() < until) {
    drive(-BASE, -BASE);
    float f = readCm(F_TRIG, F_ECHO);  delay(10);
    float l = readCm(L_TRIG, L_ECHO);  delay(10);
    float r = readCm(R_TRIG, R_ECHO);
    room(l, roomL, missL);
    room(r, roomR, missR);
    logSample(l, f, r);
  }
  drive(0, 0);
  delay(100);

  float left  = room(readCm(L_TRIG, L_ECHO), roomL, missL);  delay(10);
  float right = room(readCm(R_TRIG, R_ECHO), roomR, missR);
  if (left > OPEN_CM)       spin(1);
  else if (right > OPEN_CM) spin(-1);
  else { spin(1); spin(1); }             // dead end: turn around and backtrack
  forgetSides();
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
    Serial.print("Log server: "); Serial.print(WiFi.localIP()); Serial.println(":23 - waiting for maze_mapper.py");
  } else {
    Serial.println("No Wi-Fi - serial log only");
  }
  Serial.println("Waiting for GO");
}

void loop() {
  readCommands(Serial);
  if (wifiOk && !logger.connected()) {
    drive(0, 0);                      // nobody watching -> pause, but keep `running` so a reconnect resumes
    logger = server.accept();
    if (logger.connected()) logger.println(running ? "# tunnel robot reconnected - resuming"
                                                   : "# tunnel robot connected - send GO to start");
    delay(50);
    return;
  }
  if (logger.connected()) readCommands(logger);

  float left  = readCm(L_TRIG, L_ECHO);  delay(10);
  float front = readCm(F_TRIG, F_ECHO);  delay(10);
  float right = readCm(R_TRIG, R_ECHO);
  float l = room(left, roomL, missL);
  float r = room(right, roomR, missR);

  if (!running) {
    drive(0, 0);
  } else if (front > 0 && front < FRONT_TURN) {
    logSample(left, front, right);
    backUpAndTurn();
    return;
  } else if (l < SIDE_NEAR || r < SIDE_NEAR) {
    steerTowards(l < r ? -1 : 1);        // away from the closer side wall
  } else {
    drive(BASE, BASE);                   // straight ahead, no centring
  }

  logSample(left, front, right);
  delay(20);
}
