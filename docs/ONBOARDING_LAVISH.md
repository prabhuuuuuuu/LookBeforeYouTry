# Onboarding: Lavish

Welcome! This page gets you from zero to useful. Read the **Safety** section completely
before touching the arm or its power supply.

The project bible is `docs/LookBeforeYouTry-context.md`. If anything here disagrees with
it, the bible wins.

## 1. Safety rules (non-negotiable)

1. **Never power servos from** ESP32 GPIO pins, the Jetson USB 5 V rail, MCU signal
   pins, or PCA9685 signal pins. Servos get their own confirmed PSU.
2. **Do not assume 5 V.** The servo voltage is unconfirmed (OQ-3). Don't power anything
   until the software lead has confirmed the voltage.
3. Inline fuse and E-stop/rocker switch on the **positive** servo rail. Bulk capacitor
   across the rail. A single star ground at the PSU negative. **No** extra ground wire
   from the servo PSU to the Orin (USB already ties grounds).
4. The E-stop must be within reach at all times. **Test it before every session.**
5. **Torque off whenever hands are near the arm.**
6. **Support the arm by hand before cutting servo power** (it will drop).
7. Never run the arm unattended. Development speed cap 30%, stage cap 50%.
8. `tools/serial_probe.py` is the only tool that talks to the bridge, and only with the
   software lead present.

## 2. Laptop setup

You need Python 3.10 or newer and git.

```bash
git clone https://github.com/prabhuuuuuuu/LookBeforeYouTry.git
cd LookBeforeYouTry
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -e ".[laptop,dev]"
pytest                      # all tests should pass, no hardware needed
python -m lbyr.main --mock  # runs one simulated episode; read the printed FSM path
```

## 3. Jetson notes (software lead does this; FYI)

* JetPack 6, `sudo nvpmodel -m 0` and `sudo jetson_clocks` (MAXN).
* **Do not `pip install opencv-python` on the Jetson.** It replaces the CUDA build that
  ships with JetPack. Same for torch (use NVIDIA's Jetson wheels) and pyrealsense2
  (built from source). On the Jetson: `pip install -e ".[jetson,dev]"`.

## 4. The tools (in the order you'll use them)

Every tool has a long help text: `python tools/<name>.py --help`.

| Tool | What it does | Answers |
|---|---|---|
| `fps_test.py` | Lists cameras, measures **real** resolution and FPS | OQ-7, OQ-8, OQ-10 |
| `capture_checkerboard.py` | Live preview; press SPACE to save checkerboard photos from the Trinet | input for OQ-9 |
| `calibrate_fisheye.py` | Computes the Trinet's K, D from those photos and saves `data/calib/trinet_fisheye.yaml` | OQ-9 |
| `label_frames.py` | Shows wrist images, press 1-4 to sort into GRASPED / SLIPPING / EMPTY / OCCLUDED | Critic dataset |
| `serial_probe.py` | Sends one `{"type":"state"}` to the ESP32 and prints the reply. Read-only. **Software lead present.** | OQ-5, link check |

## 5. Suggested first tasks

1. Set up the laptop (section 2). Run `pytest` and the mock demo. Send the software lead a
   screenshot of the final episode report.
2. Plug in the Trinet and run `python tools/fps_test.py --list`, then measure 1280x720@30
   and 1920x1080@15. Write the results into OQ-7/OQ-8 in `docs/OPEN_QUESTIONS.md`.
3. Repeat step 2 with the RealSense plugged in at the same time (USB bandwidth check, OQ-10).
4. Print a checkerboard and glue it to stiff card. Measure the square size with calipers.
   Capture 15-25 photos (`capture_checkerboard.py`), run `calibrate_fisheye.py --preview`,
   and record the RMS for OQ-9.
5. With the hardware team: fill in OQ-1 to OQ-6 from the ROT3U documentation and
   measurements. **Do not power the servos to find out.**
6. Later: collect wrist images and label them with `label_frames.py` (target 300-600
   images, bible Section 16).

## 6. Where data goes

Everything under `data/` (frames, datasets, calibration, episode logs) is git-ignored on
purpose. Back it up separately; never commit images or model weights.
