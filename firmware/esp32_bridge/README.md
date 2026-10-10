# Moved

The ESP32-S3 / PCA9685 bridge plan is dead. The bridge firmware is
[`firmware/xiao_c6_bridge/`](../xiao_c6_bridge/Code.ino): XIAO ESP32-C6, direct LEDC PWM,
USB serial 115200, text protocol (`help` lists the commands). Bench GUI: `tools/arm_gui.py`.
See `docs/OPEN_QUESTIONS.md` for what the host still has to implement.
