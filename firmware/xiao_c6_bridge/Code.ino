#include <Arduino.h>
#include <Preferences.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// XIAO ESP32-C6, Arduino-ESP32 core 3.x. Direct LEDC PWM, one pin per joint.
// Joint order, base -> gripper:
//   1 base yaw     D0  Feetech 25 kg
//   2 shoulder     D1  Feetech 25 kg
//   3 elbow        D3  Feetech 25 kg
//   4 wrist pitch  D4  Feetech 25 kg
//   5 wrist roll   D5  MG996R
//   6 gripper      D8  MG996R      (D6 unusable, probably UART0 TX)
// External 6.0 V servo supply; its negative MUST be tied to XIAO GND.
// There is NO position feedback. "Position" = last commanded pulse.
//
// MOTION: every joint ramps up, cruises and ramps down (speed + acceleration
// limited). Joints 5-6 run slower. Backlash: every move finishes by approaching
// the target from below; a move that would arrive from above first goes
// BACKLASH_US past it, then comes back up, so the gears settle on the same face.
//
// BOOT: joints attach one at a time (staggered) at their saved zero and hold.
// The first pulse cannot be smoothed (no feedback): rest the arm in its zero
// pose before powering the servo rail.
//
// Serial 115200, newline-terminated:
//   J              test joint J: swing +/-30 deg, back to zero
//   g J DEG        move joint J to DEG relative to its zero
//   m D1 .. D6     move all joints together, arriving at the same time
//   t J D          nudge joint J by D us from its target; then 'z J' to store
//   z [J]          store the current target as zero (saved)
//   set J US       set joint J's zero to exact pulse US, move there, save
//   save           write all zeros to flash
//   home           move all joints to their zeros (attaches any released joint)
//   x | stop       brake smoothly and hold
//   speed V A      max speed deg/s and accel deg/s^2 (capped)
//   off [J]        release PWM (arm may fall; support it first)
//   list           print state;  help  print commands

const int COUNT = 6;
const uint8_t PINS[COUNT] = {D0, D1, D3, D4, D5, D8};
const char *NAMES[COUNT] = {"Joint 1 Feetech", "Joint 2 Feetech", "Joint 3 Feetech",
                            "Joint 4 Feetech", "Joint 5 MG996R",  "Joint 6 MG996R"};

const uint8_t PWM_BITS = 14;
const uint32_t PERIOD_US = 20000;          // 50 Hz
const float US_PER_DEG = 1000.0f / 90.0f;  // nominal 180 deg = 2000 us; per-joint calibration later

// Per-joint hard limits. Standard servo range for now: MEASURE each joint's
// mechanical limits on the assembled arm and tighten these.
const int MIN_US[COUNT] = {500, 500, 500, 500, 500, 500};
const int MAX_US[COUNT] = {2500, 2500, 2500, 2500, 2500, 2500};
// Fallback zeros if flash is empty (zero_positions.json, 2026-10-10 19:38:39).
const int DEFAULT_ZERO_US[COUNT] = {1477, 1989, 1844, 1223, 1467, 1433};

// Speed/accel limits; JOINT_SPEED scales them per joint (MG996R gentler).
const float VMAX_CAP_DEG = 60.0f;
const float AMAX_CAP_DEG = 200.0f;
float vmaxDeg = 30.0f;
float amaxDeg = 60.0f;
const float JOINT_SPEED[COUNT] = {1.0f, 1.0f, 1.0f, 1.0f, 0.35f, 0.35f};

const float BACKLASH_US = 20;  // ~2 deg; tune per joint if needed
const float TEST_DEG = 30;
const uint32_t BOOT_DELAY_MS = 1000;     // let the servo supply settle
const uint32_t ATTACH_STAGGER_MS = 300;  // one joint at a time limits inrush
const uint32_t TICK_US = 10000;          // planner at 100 Hz
const float SETTLE_US = 0.5f;

struct Joint {
  int zeroUs;
  float pos;     // commanded pulse now (us)
  float vel;     // us/s
  float tgt;     // current leg target (us)
  float fin;     // real target after a backlash overshoot
  bool hasFin;
  float scale;   // 0..1, slows the joint so synced moves arrive together
  bool attached;
};

Joint J[COUNT];
Preferences prefs;
uint32_t lastTick = 0;
char line[96];
size_t used = 0;

uint32_t ticks(float us) {
  return (uint32_t)((us * (1UL << PWM_BITS) + PERIOD_US / 2) / PERIOD_US);
}

float clampUs(int j, float us) {
  if (us < MIN_US[j]) return MIN_US[j];
  if (us > MAX_US[j]) return MAX_US[j];
  return us;
}

void writeJoint(int j) { ledcWrite(PINS[j], ticks(J[j].pos)); }

float goalOf(int j) { return J[j].hasFin ? J[j].fin : J[j].tgt; }

bool attachJoint(int j) {
  Joint &q = J[j];
  if (q.attached) return true;
  if (!ledcAttach(PINS[j], 50, PWM_BITS)) return false;
  q.pos = q.tgt = clampUs(j, q.zeroUs);  // first pulse = zero; servo goes there
  q.vel = 0;
  q.hasFin = false;
  q.scale = 1;
  writeJoint(j);
  q.attached = true;
  return true;
}

void releaseJoint(int j) {
  Joint &q = J[j];
  if (!q.attached) return;
  ledcWrite(PINS[j], 0);
  ledcDetach(PINS[j]);
  pinMode(PINS[j], OUTPUT);
  digitalWrite(PINS[j], LOW);
  q.vel = 0;
  q.attached = false;
}

// Where joint j should end up. Approaches from below to cancel backlash.
void setTarget(int j, float us, float scale) {
  Joint &q = J[j];
  us = clampUs(j, us);
  q.scale = scale < 0.02f ? 0.02f : (scale > 1.0f ? 1.0f : scale);
  if (us < q.pos) {
    q.tgt = clampUs(j, us - BACKLASH_US);
    q.fin = us;
    q.hasFin = true;
  } else {
    q.tgt = us;
    q.hasFin = false;
  }
}

// One planner step: accelerate toward the target, never faster than vmax,
// braking early enough (in discrete 10 ms steps) to stop exactly on target.
void stepJoint(int j, float dt) {
  Joint &q = J[j];
  float k = q.scale * JOINT_SPEED[j];
  float amax = amaxDeg * US_PER_DEG * k;
  float vmax = vmaxDeg * US_PER_DEG * k;
  float err = q.tgt - q.pos;

  if (fabsf(err) <= SETTLE_US && fabsf(q.vel) <= amax * dt) {
    if (q.pos != q.tgt || q.vel != 0) {
      q.pos = q.tgt;
      q.vel = 0;
      writeJoint(j);
    }
    if (q.hasFin) {  // overshoot leg done: come back up to the real target
      q.tgt = q.fin;
      q.hasFin = false;
    }
    return;
  }

  float h = amax * dt * dt;
  float n = (-1.0f + sqrtf(1.0f + 8.0f * fabsf(err) / h)) * 0.5f;
  float vdes = n * amax * dt;
  if (vdes > vmax) vdes = vmax;
  if (err < 0) vdes = -vdes;

  float dv = vdes - q.vel;
  float maxdv = amax * dt;
  if (dv > maxdv) dv = maxdv;
  if (dv < -maxdv) dv = -maxdv;
  q.vel += dv;

  float next = q.pos + q.vel * dt;
  if ((err > 0 && next > q.tgt) || (err < 0 && next < q.tgt)) {
    next = q.tgt;
    q.vel = 0;
  }
  q.pos = clampUs(j, next);
  writeJoint(j);
}

void service() {
  uint32_t now = micros();
  uint32_t elapsed = now - lastTick;
  if (elapsed < TICK_US) return;
  lastTick = now;
  float dt = elapsed * 1e-6f;
  if (dt > 0.05f) dt = 0.05f;
  for (int j = 0; j < COUNT; j++)
    if (J[j].attached) stepJoint(j, dt);
}

bool settled(int j) { return J[j].pos == J[j].tgt && J[j].vel == 0 && !J[j].hasFin; }

void waitMs(uint32_t ms) {
  uint32_t t0 = millis();
  while (millis() - t0 < ms) { service(); delay(1); }
}

void waitSettled(int j) {
  while (!settled(j)) { service(); delay(1); }
  waitMs(300);
}

// Move several joints so they start and arrive together (from rest).
void moveSynced(const float *goalUs, const bool *use) {
  float d[COUNT];
  float dmax = 0;
  for (int j = 0; j < COUNT; j++) {
    d[j] = use[j] ? fabsf(clampUs(j, goalUs[j]) - J[j].pos) : 0;
    if (d[j] > dmax) dmax = d[j];
  }
  for (int j = 0; j < COUNT; j++)
    if (use[j]) setTarget(j, goalUs[j], dmax > 0 ? d[j] / dmax : 1.0f);
}

void stopAll() {  // brake at the acceleration limit, then hold
  for (int j = 0; j < COUNT; j++) {
    Joint &q = J[j];
    if (!q.attached) continue;
    float amax = amaxDeg * US_PER_DEG * q.scale * JOINT_SPEED[j];
    q.tgt = clampUs(j, q.pos + q.vel * fabsf(q.vel) / (2.0f * amax));
    q.hasFin = false;
  }
}

void homeAll() {
  for (int j = 0; j < COUNT; j++) {
    if (J[j].attached) continue;
    if (!attachJoint(j)) { Serial.printf("ERR attach %d\n", j + 1); continue; }
    waitMs(ATTACH_STAGGER_MS);
  }
  float goal[COUNT];
  bool use[COUNT];
  for (int j = 0; j < COUNT; j++) { goal[j] = J[j].zeroUs; use[j] = J[j].attached; }
  moveSynced(goal, use);
}

void moveDeg(int j, float deg) {
  if (!attachJoint(j)) { Serial.println("ERR attach"); return; }
  setTarget(j, J[j].zeroUs + deg * US_PER_DEG, 1.0f);
}

void storeZero(int j) {
  char key[8];
  snprintf(key, sizeof key, "z%d", j);
  prefs.putInt(key, J[j].zeroUs);
}

void list() {
  for (int j = 0; j < COUNT; j++)
    Serial.printf("S,%d,%d,%.0f,%d,%s\n", j + 1, J[j].zeroUs, J[j].pos,
                  J[j].attached ? 1 : 0, NAMES[j]);
  Serial.printf("V,%.0f,%.0f\n", vmaxDeg, amaxDeg);
}

void help() {
  Serial.println("J test | g J DEG | m D1..D6 | t J D nudge | z [J] zero | set J US | save");
  Serial.println("home | x stop | speed V A | off [J] | list | help");
  Serial.println("Boot: joints go to saved zeros. Rest arm in zero pose before power. Support arm before 'off'.");
}

void testJoint(int j) {
  Serial.printf("Testing %s (GPIO %d)\n", NAMES[j], PINS[j]);
  if (!attachJoint(j)) { Serial.println("ERR attach"); return; }
  waitSettled(j);
  moveDeg(j, TEST_DEG);  waitSettled(j);
  moveDeg(j, -TEST_DEG); waitSettled(j);
  moveDeg(j, 0);         waitSettled(j);
  Serial.println("OK test done (holding zero; 'off' to release)");
}

void nudge(int j, int d) {
  if (!attachJoint(j)) { Serial.println("ERR attach"); return; }
  setTarget(j, goalOf(j) + d, 1.0f);
  Serial.printf("OK joint %d target=%.0f us (zero is still %d; 'z %d' to store)\n",
                j + 1, goalOf(j), J[j].zeroUs, j + 1);
}

void setZero(int j, int us) {
  if (!attachJoint(j)) { Serial.println("ERR attach"); return; }
  J[j].zeroUs = (int)clampUs(j, (float)us);
  storeZero(j);
  setTarget(j, J[j].zeroUs, 1.0f);
  Serial.printf("OK joint %d zero set to %d us (saved)\n", j + 1, J[j].zeroUs);
}

void zeroJoint(int j) {
  if (!J[j].attached) { Serial.printf("Joint %d not active; nothing to store\n", j + 1); return; }
  J[j].zeroUs = (int)(goalOf(j) + 0.5f);
  storeZero(j);
  Serial.printf("OK zero joint %d = %d us (saved)\n", j + 1, J[j].zeroUs);
}

bool validJoint(int j) { return j >= 1 && j <= COUNT; }

void command(char *cmd) {
  int j, d;
  float v, a;
  float m[COUNT];
  char extra;
  if (!strcmp(cmd, "help")) { help(); return; }
  if (!strcmp(cmd, "list")) { list(); return; }
  if (!strcmp(cmd, "save")) { for (int i = 0; i < COUNT; i++) storeZero(i); Serial.println("OK saved all zeros"); return; }
  if (!strcmp(cmd, "home")) { homeAll(); Serial.println("OK home"); return; }
  if (!strcmp(cmd, "x") || !strcmp(cmd, "stop")) { stopAll(); Serial.println("OK stopped; holding"); return; }
  if (!strcmp(cmd, "off")) { for (int i = 0; i < COUNT; i++) releaseJoint(i); Serial.println("OK off"); return; }
  if (sscanf(cmd, "off %d %c", &j, &extra) == 1 && validJoint(j)) {
    releaseJoint(j - 1); Serial.printf("OK off %d\n", j); return;
  }
  if (!strcmp(cmd, "z")) { for (int i = 0; i < COUNT; i++) zeroJoint(i); return; }
  if (sscanf(cmd, "z %d %c", &j, &extra) == 1 && validJoint(j)) { zeroJoint(j - 1); return; }
  if (sscanf(cmd, "t %d %d %c", &j, &d, &extra) == 2 && validJoint(j)) { nudge(j - 1, d); return; }
  if (sscanf(cmd, "set %d %d %c", &j, &d, &extra) == 2 && validJoint(j)) { setZero(j - 1, d); return; }
  if (sscanf(cmd, "speed %f %f %c", &v, &a, &extra) == 2) {
    if (v <= 0 || a <= 0) { Serial.println("ERR speed must be > 0"); return; }
    vmaxDeg = v > VMAX_CAP_DEG ? VMAX_CAP_DEG : v;
    amaxDeg = a > AMAX_CAP_DEG ? AMAX_CAP_DEG : a;
    Serial.printf("OK speed %.0f deg/s, %.0f deg/s^2\n", vmaxDeg, amaxDeg);
    return;
  }
  if (sscanf(cmd, "m %f %f %f %f %f %f %c", &m[0], &m[1], &m[2], &m[3], &m[4], &m[5],
             &extra) == COUNT) {
    float goal[COUNT];
    bool use[COUNT];
    for (int i = 0; i < COUNT; i++) {
      use[i] = attachJoint(i);
      goal[i] = J[i].zeroUs + m[i] * US_PER_DEG;
    }
    moveSynced(goal, use);
    Serial.println("OK m");
    return;
  }
  if (sscanf(cmd, "g %d %f %c", &j, &v, &extra) == 2 && validJoint(j)) {
    moveDeg(j - 1, v);
    Serial.printf("OK g %d %.1f deg\n", j, v);
    return;
  }
  if (sscanf(cmd, "%d %c", &j, &extra) == 1 && validJoint(j)) { testJoint(j - 1); return; }
  Serial.println("ERR unknown command; enter help");
}

void setup() {
  for (int i = 0; i < COUNT; i++) {
    digitalWrite(PINS[i], LOW);
    pinMode(PINS[i], OUTPUT);
  }
  Serial.begin(115200);
  delay(BOOT_DELAY_MS);
  prefs.begin("arm", false);
  for (int i = 0; i < COUNT; i++) {
    char key[8];
    snprintf(key, sizeof key, "z%d", i);
    J[i].zeroUs = prefs.getInt(key, DEFAULT_ZERO_US[i]);
    J[i].attached = false;
    J[i].hasFin = false;
    J[i].vel = 0;
    J[i].scale = 1;
  }
  homeAll();  // staggered attach at saved zeros, then hold
  help();
}

void loop() {
  service();
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') {
      if (used) { line[used] = '\0'; used = 0; command(line); }
    } else if (used < sizeof(line) - 1) {
      line[used++] = c;
    }
  }
}
