// Host harness: runs setup() then loop() every simulated millisecond. Script on stdin:
// "<t_ms> <command>" lines (sorted); the run ends at the last "<t_ms> END".
#include "Arduino.h"
uint64_t g_us = 0;
std::string g_rx;
SerialT Serial;
#include "Code.ino"
#include <iostream>
#include <vector>

int main() {
  std::vector<std::pair<uint64_t, std::string>> script;
  std::string ln;
  while (std::getline(std::cin, ln)) {
    size_t sp = ln.find(' ');
    if (sp == std::string::npos) continue;
    script.push_back({std::stoull(ln.substr(0, sp)) * 1000, ln.substr(sp + 1)});
  }
  setup();
  size_t k = 0;
  while (k < script.size()) {
    while (k < script.size() && script[k].first <= g_us) {
      if (script[k].second == "END") return 0;
      g_rx += script[k].second + "\n";
      k++;
    }
    g_us += 1000;
    loop();
  }
  return 0;
}
