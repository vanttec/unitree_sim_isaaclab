"""Record teleoperated loco-manipulation on the REAL G1 (read-only: subscribes, never publishes).

Legs: Unitree's own locomotion (Regular mode, R3 remote). Arms/hands: xr_teleoperate --motion --ee brainco.
Same episode layout as the sim recorder (episode.npz + motion_50hz.csv + meta.json), so
`python -m locomanip.export` works on both. The pelvis pose is the robot's odometry estimate
(rt/odommodestate), not ground truth: it drifts over long episodes.

    cd ~/unitree_sim_isaaclab && conda activate tv
    python -m locomanip.real_recorder --iface enx0c37969f434c --label pick_box
    keys:  r = start   f = stop/save   g = discard   q = quit

Topics (domain 0):
    rt/lowstate              29 joints q/dq/tau, IMU, wireless_remote (R3 sticks + buttons)
    rt/odommodestate         pelvis position / velocity / orientation estimate (500 Hz)
    rt/arm_sdk               arm targets xr_teleoperate sends in --motion mode (+ blend weight in slot 29)
    rt/brainco/{left,right}/{state,cmd}   Revo2 fingers, normalized 0..1
                             order [thumb, thumb_aux, index, middle, ring, pinky]
Head camera: teleimager's ZMQ JPEG stream (tcp://127.0.0.1:55555 on the Jetson), stored as received:
    head_cam.mjpeg           concatenated JPEGs (ffmpeg -f mjpeg -i head_cam.mjpeg out.mp4)
    head_cam_index.npz       t_wall per frame (same clock as episode.npz) + byte offset/length
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np
import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_, SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from .g1_joints import ARM_LOWCMD_SLICE, BODY_JOINTS_MJ
from .recorder import CSV_FPS, write_bones_csv

HZ = 50.0
REVO2_RANGE_DEG = [59, 90, 81, 81, 81, 81]   # thumb flex, thumb aux, index, middle, ring, pinky (BrainCo Revo2 spec)


def parse_remote(b):
    """Unitree xKeySwitchUnion layout (same as ~/g1_deploy/deploy/wireless_remote.py on the Jetson)."""
    b = bytes(int(x) & 0xFF for x in b)
    if len(b) < 24:
        return np.full(5, np.nan, np.float32), 0
    keys = struct.unpack_from("<H", b, 2)[0]
    lx, rx, ry, l2, ly = struct.unpack_from("<5f", b, 4)
    return np.array([lx, ly, rx, ry, l2], np.float32), keys


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.msg = {}
        self.count = {}

    def cb(self, name):
        def f(m):
            with self.lock:
                self.msg[name] = m
                self.count[name] = self.count.get(name, 0) + 1
        return f

    def get(self, name):
        with self.lock:
            return self.msg.get(name)


def snapshot(L: Latest):
    ls, od, arm = L.get("lowstate"), L.get("odom"), L.get("arm_sdk")
    if ls is None or od is None:
        return None
    q = np.array([ls.motor_state[i].q for i in range(29)], np.float32)
    dq = np.array([ls.motor_state[i].dq for i in range(29)], np.float32)
    tau = np.array([ls.motor_state[i].tau_est for i in range(29)], np.float32)
    remote, keys = parse_remote(ls.wireless_remote)
    s = {
        "root_pos": np.array(od.position, np.float32),
        "root_quat_wxyz": np.array(od.imu_state.quaternion, np.float32),
        "root_lin_vel": np.array(od.velocity, np.float32),
        "yaw_speed": np.float32(od.yaw_speed),
        "imu_quat_wxyz": np.array(ls.imu_state.quaternion, np.float32),
        "imu_gyro": np.array(ls.imu_state.gyroscope, np.float32),
        "imu_acc": np.array(ls.imu_state.accelerometer, np.float32),
        "q": q, "dq": dq, "tau": tau,
        "remote": remote, "remote_keys": np.uint16(keys),
        "cmd_arm_q": np.full(14, np.nan, np.float32), "arm_sdk_weight": np.float32(np.nan),
    }
    if arm is not None:
        s["cmd_arm_q"] = np.array([arm.motor_cmd[i].q for i in range(29)], np.float32)[ARM_LOWCMD_SLICE]
        s["arm_sdk_weight"] = np.float32(arm.motor_cmd[29].q)
    for side in ("left", "right"):
        for kind in ("state", "cmd"):
            m = L.get(f"{side}_{kind}")
            arr = np.full(6, np.nan, np.float32)
            if m is not None:
                items = m.states if kind == "state" else m.cmds
                arr[:min(6, len(items))] = [items[i].q for i in range(min(6, len(items)))]
            s[f"hand_{side}_{kind}"] = arr
    return s


class HeadCam(threading.Thread):
    """Keeps every JPEG teleimager publishes while recording (no decode: the Jetson CPU is busy enough)."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.lock = url, threading.Lock()
        self.rec, self.frames, self.t, self.seen, self.running = False, [], [], 0, True

    def run(self):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 60)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.url)
        while self.running:
            try:
                b = sock.recv()
            except zmq.Again:
                continue
            now = time.time()
            with self.lock:
                self.seen += 1
                if self.rec:
                    self.frames.append(b)
                    self.t.append(now)
        sock.close(0)

    def start_rec(self):
        with self.lock:
            self.rec, self.frames, self.t = True, [], []

    def stop_rec(self):
        with self.lock:
            self.rec = False
            out, self.frames, self.t = (self.frames, self.t), [], []
        return out


def save_cam(ep, frames, t, t0):
    if not frames:
        print("\r[rec] head cam: no frames (is teleimager.service running?)")
        return 0, 0.0
    lens = np.array([len(f) for f in frames], np.int64)
    with open(os.path.join(ep, "head_cam.mjpeg"), "wb") as f:
        for b in frames:
            f.write(b)
    t = np.asarray(t, np.float64)
    np.savez(os.path.join(ep, "head_cam_index.npz"), t_wall=t, t_rel=t - t0,
             offset=np.concatenate([[0], np.cumsum(lens)[:-1]]), length=lens)
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0
    return len(frames), fps


def save(out_dir, label, rows, t, cam=None):
    t = np.asarray(t, np.float64)
    arr = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ep = os.path.join(out_dir, f"{stamp}_{label}")
    os.makedirs(ep, exist_ok=True)
    np.savez_compressed(os.path.join(ep, "episode.npz"), t_wall=t, **arr, body_joint_names=np.array(BODY_JOINTS_MJ))
    tt = t - t[0]
    n_img, cam_fps = save_cam(ep, *cam, t[0]) if cam is not None else (0, 0.0)
    frames, offset = write_bones_csv(os.path.join(ep, "motion_50hz.csv"), tt, arr["root_pos"], arr["root_quat_wxyz"], arr["q"], 0.0)
    meta = {"task": "real_g1_teleop", "label": label, "steps": len(t), "duration_sim_s": float(tt[-1]),
            "duration_wall_s": float(tt[-1]), "csv_fps": CSV_FPS, "csv_frames": frames, "root_xy_offset_m": offset,
            "ground_z_m": 0.0, "body_joint_names_mj": BODY_JOINTS_MJ,
            "hand": "BrainCo Revo2, normalized 0..1 (0 = open, 1 = closed) [thumb, thumb_aux, index, middle, ring, pinky]",
            "hand_range_deg": REVO2_RANGE_DEG, "remote": "[lx, ly, rx, ry, L2] + remote_keys bitfield",
            "source": "real G1: rt/odommodestate estimate + rt/lowstate",
            "head_cam": {"frames": n_img, "fps": round(cam_fps, 2), "format": "JPEG 640x480 from teleimager (head RealSense)"}
            if cam is not None else None}
    with open(os.path.join(ep, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    dist = float(np.linalg.norm(arr["root_pos"][-1, :2] - arr["root_pos"][0, :2]))
    cam_txt = f", head cam {n_img} imgs @ {cam_fps:.1f} fps" if cam is not None else ""
    print(f"\r[rec] saved '{label}': {tt[-1]:.1f} s, {frames} frames{cam_txt}, pelvis moved {dist:.2f} m -> {ep}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="NIC on the robot network, e.g. enx0c37969f434c")
    p.add_argument("--label", default="episode")
    p.add_argument("--out", default="data/locomanip_real")
    p.add_argument("--cam", default="tcp://127.0.0.1:55555", help="teleimager head-camera ZMQ stream")
    p.add_argument("--no-cam", action="store_true", help="do not record the head camera")
    a = p.parse_args()
    cam = None if a.no_cam else HeadCam(a.cam)
    if cam is not None:
        cam.start()

    ChannelFactoryInitialize(0, a.iface)
    L = Latest()
    subs = []
    for name, topic, typ in [("lowstate", "rt/lowstate", LowState_), ("odom", "rt/odommodestate", SportModeState_),
                             ("arm_sdk", "rt/arm_sdk", LowCmd_),
                             ("left_state", "rt/brainco/left/state", MotorStates_), ("right_state", "rt/brainco/right/state", MotorStates_),
                             ("left_cmd", "rt/brainco/left/cmd", MotorCmds_), ("right_cmd", "rt/brainco/right/cmd", MotorCmds_)]:
        s = ChannelSubscriber(topic, typ)
        s.Init(L.cb(name), 10)
        subs.append(s)

    time.sleep(1.5)
    print("topics seen (1.5 s):", {k: v for k, v in L.count.items()},
          f"| head cam frames: {cam.seen}" if cam is not None else "| head cam: off")
    if cam is not None and cam.seen == 0:
        print(f"!! no head-camera frames from {a.cam}: check teleimager.service (or use --no-cam)")
    for need in ("lowstate", "odom"):
        if need not in L.count:
            print(f"!! no {need}: check the cable/iface ({a.iface}) and that the robot is on")
    print("r = start   f = stop/save   g = discard   q = quit")

    state = {"rec": False, "rows": [], "t": [], "n": 0, "quit": False}
    lock = threading.Lock()

    def sampler():
        period = 1.0 / HZ
        nxt = time.perf_counter()
        while not state["quit"]:
            nxt += period
            s = snapshot(L)
            with lock:
                if state["rec"] and s is not None:
                    state["rows"].append(s)
                    state["t"].append(time.time())
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def finish(keep):
        """Stop the current episode (caller holds `lock`) and save it with its camera frames."""
        state["rec"] = False
        frames = cam.stop_rec() if cam is not None else None
        if not keep:
            return
        if len(state["t"]) > 10:
            save(a.out, state["label"], state["rows"], state["t"], frames)
        else:
            print("[rec] too short, not saved")

    threading.Thread(target=sampler, daemon=True).start()
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            c = sys.stdin.read(1).lower()
            with lock:
                if c == "r" and not state["rec"]:
                    state["n"] += 1
                    state.update(rec=True, rows=[], t=[], label=f"{a.label}_{state['n']:03d}")
                    if cam is not None:
                        cam.start_rec()
                    print(f"[rec] START {state['label']}")
                elif c == "f" and state["rec"]:
                    finish(keep=True)
                elif c == "g" and state["rec"]:
                    finish(keep=False)
                    print(f"[rec] discarded {state['label']}")
                elif c == "q":
                    if state["rec"]:
                        finish(keep=True)
                    break
    except KeyboardInterrupt:
        with lock:
            if state["rec"]:
                finish(keep=True)
    finally:
        state["quit"] = True
        if cam is not None:
            cam.running = False
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
