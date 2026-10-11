/*
  L tunnel, driven over Wi-Fi and mapped by maze_mapper.py.
    - drive forward centred between the walls
    - one side open (the corner is coming) -> keep the other wall at half the tunnel width
    - wall ahead -> spin towards the open side in short pulses until the front is clear
      and the old front wall is on the other side sensor (= turned ~90 deg)
    - after the turn, both sides open for a while -> out of the tunnel, stop (logs EXIT)

  It waits for "GO" (Space in maze_mapper.py, or type GO in the Serial Monitor). "STOP" stops it.
  Over Wi-Fi it pauses while the mapper is disconnected and carries on by itself when it
  reconnects (unless STOP was sent). No Wi-Fi -> it starts by itself START_MS after power-up.
  Log line: S,<millis>,<left_cm>,<front_cm>,<right_cm>,<pwm_left>,<pwm_right>   (-1 = no echo)
  Turn line: T,<millis>,<way>,<ms>   (way 1 = spun left, -1 = right, for <ms> ms) - one per pulse,
  so set the mapper's --spin-rate to how many degrees one pulse turns / (PULSE_MS / 1000).
*/
#include <WiFiS3.h>
#include "arduino_secrets.h"   // copy arduino_secrets.example.h to this name and fill in your Wi-Fi

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
const int   START_MS   = 3000; // no Wi-Fi: wait this long after power-up, then go
const int   BASE       = 60;   // forward speed (PWM 0-255)
const int   KICK       = 120;  // short burst to get going from a stop...
const int   KICK_MS    = 60;   // ...for this long
const float KP         = -4;   // steering strength; negative because left/right turned out swapped on the robot
const int   MAX_STEER  = 30;   // never steer harder than this
const float FRONT_CM   = 15;   // wall ahead closer than this -> turn
const float OPEN_CM    = 35;   // a side further than this (or no echo) is open
const int   PULSE_PWM  = 130;  // spin pulse strength
const int   PULSE_MS   = 90;   // spin pulse length
const int   MAX_PULSES = 50;   // turn taking more than this -> give up and stop
const int   EXIT_N     = 15;   // after the turn: this many loops with both sides open -> out, stop
const int   SENSOR_SIDE = 1;   // -1 if the L sensor really faces the robot's right (then run the mapper with --swap-sides)
const float NO_ECHO    = 999;

float halfW = 12;              // half the tunnel width, learned while both walls are seen
bool  turned = false;
int   bothOpen = 0;
int   pwmL = 0, pwmR = 0;

WiFiServer server(23);         // log port
WiFiClient logger;
bool wifiOk = false;
bool running = false;          // set by GO, cleared by STOP / exit / giving up
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

float readCm(int trig, int echo) {   // NO_ECHO if nothing came back
  digitalWrite(trig, LOW);  delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 12000);
  return us == 0 ? NO_ECHO : us * 0.0343 / 2;
}

void logLine(String line) {
  Serial.println(line);
  if (logger.connected()) logger.println(line);
}

String cm(float d) { return d >= NO_ECHO ? String("-1") : String(d, 1); }

void logSample(float left, float front, float right) {
  logLine("S," + String(millis()) + "," + cm(left) + "," + cm(front) + "," + cm(right) + "," +
      String(pwmL) + "," + String(pwmR));
}

void kick() {
  drive(KICK, KICK);
  delay(KICK_MS);
}

void finish(String why) {            // out of the tunnel (or gave up): stop until the next GO
  drive(0, 0);
  running = false;
  turned = false;
  bothOpen = 0;
  logLine(why);
}

void handleCommand(String c) {
  c.trim();
  c.toUpperCase();
  if (c == "GO") {
    if (!running) kick();
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

bool keepGoing() {                   // still allowed to move? (STOP sent / mapper gone -> no)
  readCommands(Serial);
  if (logger.connected()) readCommands(logger);
  return running && (!wifiOk || logger.connected());
}

void pulse(int dir) {                // one short spin; dir 1 = towards the L sensor's side; logged as a T line
  drive(PULSE_PWM * dir, -PULSE_PWM * dir);
  delay(PULSE_MS);
  drive(0, 0);
  logLine("T," + String(millis()) + "," + String(dir * SENSOR_SIDE) + "," + String(PULSE_MS));
  delay(120);                        // settle before pinging
}

// Spin towards dir until the front is clear and the wall that was ahead is now on the other side.
void turn(int dir) {
  drive(0, 0);
  delay(200);
  int clear = 0;
  for (int n = 0; n < MAX_PULSES; n++) {
    if (!keepGoing()) { drive(0, 0); return; }   // carries on from here on the next GO
    pulse(dir);
    float f = readCm(F_TRIG, F_ECHO);  delay(30);
    float l = readCm(L_TRIG, L_ECHO);  delay(10);
    float r = readCm(R_TRIG, R_ECHO);
    logSample(l, f, r);
    float s = dir > 0 ? r : l;
    clear = (f > OPEN_CM && s < OPEN_CM) ? clear + 1 : 0;
    if (clear >= 2) {
      turned = true;
      kick();
      return;
    }
  }
  finish("# couldn't find the way out of the corner - stopped");
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
    Serial.println("Waiting for GO");
  } else {
    Serial.println("No Wi-Fi - serial log only, starting by itself");
    delay(START_MS);
    running = true;
    kick();
  }
}

void loop() {
  readCommands(Serial);
  if (wifiOk && !logger.connected()) {
    drive(0, 0);                      // nobody watching -> pause, but keep `running` so a reconnect resumes
    logger = server.accept();
    if (logger.connected()) {
      logger.println(running ? "# tunnel robot reconnected - resuming"
                             : "# tunnel robot connected - send GO to start");
      if (running) kick();
    }
    delay(50);
    return;
  }
  if (logger.connected()) readCommands(logger);

  float left  = readCm(L_TRIG, L_ECHO);  delay(10);
  float front = readCm(F_TRIG, F_ECHO);  delay(10);
  float right = readCm(R_TRIG, R_ECHO);

  if (!running) {
    drive(0, 0);
    logSample(left, front, right);
    delay(50);
    return;
  }

  bool lOk = left  < OPEN_CM;
  bool rOk = right < OPEN_CM;

  if (front < FRONT_CM) {
    float l2 = readCm(L_TRIG, L_ECHO);  delay(10);   // stopped: look again at the sides
    float r2 = readCm(R_TRIG, R_ECHO);
    drive(0, 0);
    logSample(l2, front, r2);
    turn(l2 > r2 ? 1 : -1);          // towards the more open side
    return;
  }

  float diff;                        // L - R
  if (lOk && rOk) {
    halfW = 0.9 * halfW + 0.1 * (left + right) / 2;
    diff = left - right;
  } else if (lOk) {
    diff = 2 * (left - halfW);       // only the left wall: pretend the right one is still there
  } else if (rOk) {
    diff = 2 * (halfW - right);
  } else {
    diff = 0;                        // nothing either side: straight
  }

  bothOpen = (turned && !lOk && !rOk) ? bothOpen + 1 : 0;
  if (bothOpen >= EXIT_N) {          // out of the tunnel
    logSample(left, front, right);
    finish("EXIT");
    return;
  }

  int steer = constrain(KP * diff, -MAX_STEER, MAX_STEER);
  drive(BASE - steer, BASE + steer);
  logSample(left, front, right);
  delay(20);
}
