"""Slider GUI for the 6-joint arm (base -> gripper). Needs: pip install pyserial

Safety: E-stop within reach. Rest the arm in its ZERO pose before powering the servos
(the board moves every joint to its saved zero at boot).
Support the arm by hand before "Release".

Teaching poses (taught-pose mode, data/calib/poses.yaml):
    1. Move the arm with the sliders until it is where you want it.
    2. Press "Save pose as..." and type a name, for example zone_A_pre, zone_A_grasp,
       place_pre, place or home. The six slider values are saved (zero-relative degrees).
    3. "Go to pose" moves the arm to a saved pose with one 'm' command.
"""

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog, ttk

import serial
import serial.tools.list_ports
import yaml

N = 6
RANGE = 60  # slider range in degrees either side of zero (firmware clamps pulses too)
US_PER_DEG = 1000 / 90
SEND_PERIOD_S = 0.05  # at most one command per joint every 50 ms
NAMES = ["1 Base yaw", "2 Shoulder", "3 Elbow", "4 Wrist pitch", "5 Wrist roll", "6 Gripper"]


POSES_FILE = Path("data/calib/poses.yaml")


class App:
    def __init__(self, root, poses_file=POSES_FILE):
        self.root = root
        self.poses_file = poses_file
        self.ser = None
        self.rx = ""
        self.last_sent = [0.0] * N
        self.pending = [None] * N  # scheduled sends, so the final slider value is never lost
        self.mute_until = 0.0  # ignore slider callbacks while syncing from the firmware
        self.vars = [tk.DoubleVar(value=0) for _ in range(N)]
        root.title("Robot arm")

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        self.port = ttk.Combobox(top, width=14, values=self.ports())
        self.port.pack(side="left")
        if self.port["values"]:
            self.port.current(0)
        for text, cmd in [
            ("Connect", self.connect),
            ("Home", self.home),
            ("STOP", lambda: self.send("x")),
            ("Move all (synced)", self.move_all),
            ("Sync sliders", lambda: self.send("list")),
        ]:
            ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2)

        row = ttk.Frame(root, padding=(6, 0))
        row.pack(fill="x")
        ttk.Button(row, text="Set zero ALL", command=lambda: self.zero(None)).pack(side="left")
        ttk.Button(row, text="Save zeros", command=lambda: self.send("save")).pack(
            side="left", padx=2
        )
        ttk.Button(row, text="Release ALL", command=lambda: self.release(None)).pack(
            side="left", padx=2
        )
        ttk.Label(row, text="   deg/s").pack(side="left")
        self.vmax = tk.IntVar(value=30)
        ttk.Spinbox(row, from_=5, to=60, width=4, textvariable=self.vmax).pack(side="left")
        ttk.Label(row, text=" deg/s²").pack(side="left")
        self.amax = tk.IntVar(value=60)
        ttk.Spinbox(row, from_=10, to=200, width=4, textvariable=self.amax).pack(side="left")
        ttk.Button(row, text="Set speed", command=self.set_speed).pack(side="left", padx=2)

        # Taught poses: save the slider values under a name, or move to a saved pose.
        poses = ttk.Frame(root, padding=(6, 3))
        poses.pack(fill="x")
        ttk.Button(poses, text="Save pose as...", command=self.save_pose).pack(side="left")
        ttk.Label(poses, text="   Pose:").pack(side="left")
        self.pose_box = ttk.Combobox(poses, width=16, state="readonly")
        self.pose_box.pack(side="left", padx=2)
        ttk.Button(poses, text="Go to pose", command=self.go_to_pose).pack(side="left")
        self.refresh_poses()

        for j in range(N):
            f = ttk.Frame(root, padding=(6, 3))
            f.pack(fill="x")
            ttk.Label(f, text=NAMES[j], width=13).pack(side="left")
            tk.Scale(
                f,
                from_=-RANGE,
                to=RANGE,
                orient="horizontal",
                length=360,
                resolution=1,
                variable=self.vars[j],
                command=lambda v, j=j: self.slide(j),
            ).pack(side="left")
            ttk.Button(f, text="Test", width=5, command=lambda j=j: self.send(f"{j + 1}")).pack(
                side="left", padx=2
            )
            ttk.Button(f, text="Set zero", command=lambda j=j: self.zero(j)).pack(side="left")
            ttk.Button(f, text="-5us", width=5, command=lambda j=j: self.nudge(j, -5)).pack(
                side="left", padx=(2, 0)
            )
            ttk.Button(f, text="+5us", width=5, command=lambda j=j: self.nudge(j, 5)).pack(
                side="left", padx=(0, 2)
            )
            ttk.Button(f, text="Release", command=lambda j=j: self.release(j)).pack(side="left")

        self.log = tk.Text(root, height=10, width=90)
        self.log.pack(padx=6, pady=6)
        self.poll()

    def ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        try:
            self.ser = serial.Serial(self.port.get(), 115200, timeout=0)
            self.say(f"connected {self.port.get()}")
        except Exception as e:
            self.say(f"connect failed: {e}")

    def send(self, text):
        if not self.ser:
            self.say("not connected")
            return
        self.ser.write((text + "\n").encode())

    # Sliders: throttle, but always send the latest value once dragging stops.
    def slide(self, j):
        if time.time() < self.mute_until or not self.ser:
            return
        if self.pending[j] is None:
            wait = max(0.0, SEND_PERIOD_S - (time.time() - self.last_sent[j]))
            self.pending[j] = self.root.after(int(wait * 1000), lambda j=j: self.flush(j))

    def flush(self, j):
        self.pending[j] = None
        self.last_sent[j] = time.time()
        self.send(f"g {j + 1} {self.vars[j].get():.0f}")

    def set_sliders(self, values):
        self.mute_until = time.time() + 0.3  # don't echo these changes back to the arm
        for j, v in values.items():
            self.vars[j].set(v)

    def home(self):
        self.send("home")  # attaches released joints one at a time, then moves to zero
        self.set_sliders({i: 0 for i in range(N)})

    def nudge(self, j, d):
        self.send(f"t {j + 1} {d}")  # fine-tune the target; 'Set zero' stores it

    def move_all(self):
        self.send("m " + " ".join(f"{v.get():.0f}" for v in self.vars))

    def zero(self, j):
        # Current position becomes zero; sliders go back to 0 without moving anything.
        self.send(f"z {j + 1}" if j is not None else "z")
        self.set_sliders({i: 0 for i in ([j] if j is not None else range(N))})

    def release(self, j):
        self.send(f"off {j + 1}" if j is not None else "off")

    def set_speed(self):
        self.send(f"speed {self.vmax.get()} {self.amax.get()}")

    # Taught poses ----------------------------------------------------------------
    def load_poses(self):
        if not self.poses_file.is_file():
            return {}
        data = yaml.safe_load(self.poses_file.read_text()) or {}
        return data.get("poses") or {}

    def refresh_poses(self):
        names = sorted(self.load_poses())
        self.pose_box["values"] = names
        if names and not self.pose_box.get():
            self.pose_box.current(0)

    def save_pose(self):
        from lbyr.motion.taught import save_pose  # same file format as the robot code

        name = simpledialog.askstring(
            "Save pose", "Pose name (e.g. zone_A_pre, zone_A_grasp, place_pre, place, home):"
        )
        if not name:
            return
        values = [round(v.get()) for v in self.vars]
        try:
            save_pose(self.poses_file, name.strip(), values)
        except ValueError as e:
            self.say(f"not saved: {e}")
            return
        self.say(f"saved pose {name.strip()} = {values} to {self.poses_file}")
        self.refresh_poses()
        self.pose_box.set(name.strip())

    def go_to_pose(self):
        name = self.pose_box.get()
        values = self.load_poses().get(name)
        if not values:
            self.say("choose a saved pose first")
            return
        self.send("m " + " ".join(f"{float(v):.1f}" for v in values))  # one synced move
        self.set_sliders({j: float(v) for j, v in enumerate(values)})
        self.say(f"going to pose {name}")

    def say(self, s):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def handle_line(self, text):
        self.say(text)
        parts = text.split(",")
        if parts[0] != "S" or len(parts) < 5:
            return
        try:
            j, zero, cur, attached = (
                int(parts[1]) - 1,
                int(parts[2]),
                float(parts[3]),
                parts[4] == "1",
            )
        except ValueError:
            return
        if attached and 0 <= j < N:
            self.set_sliders({j: round((cur - zero) / US_PER_DEG)})

    def poll(self):
        if self.ser and self.ser.in_waiting:
            self.rx += self.ser.read(self.ser.in_waiting).decode(errors="replace")
            *lines, self.rx = self.rx.split("\n")
            for text in lines:
                if text.strip():
                    self.handle_line(text.strip())
        self.root.after(50, self.poll)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--poses", type=Path, default=POSES_FILE, help="taught poses file")
    args = p.parse_args()
    root = tk.Tk()
    App(root, args.poses)
    root.mainloop()


if __name__ == "__main__":
    main()
