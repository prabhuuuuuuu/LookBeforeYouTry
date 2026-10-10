// Minimal Arduino stub for running firmware/xiao_c6_bridge/Code.ino on the host
// (tests/test_fw_sim.py). Simulated clock; Serial goes to stdout with a timestamp.
#pragma once
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>

enum { D0, D1, D2, D3, D4, D5, D6, D7, D8 };
#define OUTPUT 1
#define LOW 0

extern uint64_t g_us;
extern std::string g_rx;

struct SerialT {
  std::string out;
  void begin(long) {}
  int available() { return (int)g_rx.size(); }
  int read() { char c = g_rx[0]; g_rx.erase(0, 1); return (unsigned char)c; }
  void emit(const char *s) {
    for (const char *p = s; *p; p++) {
      if (*p == '\n') { ::printf("%llu|%s\n", (unsigned long long)(g_us / 1000), out.c_str()); out.clear(); }
      else out += *p;
    }
  }
  void println(const char *s) { emit(s); emit("\n"); }
  int printf(const char *fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt); int n = vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    emit(buf);
    return n;
  }
};
extern SerialT Serial;
inline bool ledcAttach(uint8_t, uint32_t, uint8_t) { return true; }
inline void ledcWrite(uint8_t, uint32_t) {}
inline void ledcDetach(uint8_t) {}
inline void pinMode(uint8_t, int) {}
inline void digitalWrite(uint8_t, int) {}
inline uint32_t micros() { return (uint32_t)g_us; }
inline uint32_t millis() { return (uint32_t)(g_us / 1000); }
inline void delay(uint32_t ms) { g_us += (uint64_t)ms * 1000; }
