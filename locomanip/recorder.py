"""In-process recorder for teleoperated loco-manipulation in Isaac Lab.

Hooked into sim_main.py's main loop (one call per control step). It reads ground truth straight
from the scene -- pelvis pose/velocity, all joint states, rigid objects -- plus what the operator
asked for (rt/run_command/cmd velocity, rt/lowcmd arm targets, rt/inspire/cmd hand targets),
so an episode carries both "what was commanded" and "what the robot did".

Episodes are toggled over DDS (rt/locomanip/record_cmd, see cli.py):
    "start <label>"   begin a new episode
    "stop"            save it
    "discard"         drop it

Each saved episode directory holds
    episode.npz       every control step, raw (sim time, 100 Hz nominal)
    motion_50hz.csv   Bones-SEED format resampled to 50 Hz: what SONIC's converter ingests
    meta.json         task, joint names, counts, root offset
"""
from __future__ import annotations

import json
import os
import queue
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .g1_joints import ARM_LOWCMD_SLICE, BODY_JOINTS_MJ, RECORD_TOPIC

CSV_FPS = 50


def _np(t):
    return t.detach().cpu().numpy()


def _parse_run_command(raw):
    """rt/run_command/cmd carries "[vx, vy, wz, height]" as a string (or a list)."""
    if not raw or "run_command" not in raw:
        return np.full(4, np.nan, np.float32)
    v = raw["run_command"]
    try:
        if isinstance(v, str):
            v = json.loads(v.replace("(", "[").replace(")", "]"))
        return np.asarray(v[:4], np.float32)
    except Exception:
        return np.full(4, np.nan, np.float32)


class LocomanipRecorder:
    def __init__(self, env, task_name: str, out_dir: str, dds_manager=None, ground_z: float = 0.0):
        self.env = env
        self.task = task_name
        self.out_dir = os.path.abspath(out_dir)
        self.ground_z = ground_z
        self.robot = env.scene["robot"]
        names = list(self.robot.data.joint_names)
        missing = [j for j in BODY_JOINTS_MJ if j not in names]
        if missing:
            raise RuntimeError(f"[locomanip] robot lacks body joints: {missing}")
        self.body_idx = np.array([names.index(j) for j in BODY_JOINTS_MJ])
        self.hand_names = [n for n in names if n not in BODY_JOINTS_MJ]
        self.hand_idx = np.array([names.index(n) for n in self.hand_names], dtype=int)
        self.objects = dict(getattr(env.scene, "rigid_objects", {}) or {})

        get = (lambda k: dds_manager.get_object(k)) if dds_manager else (lambda k: None)
        self.cmd_run, self.cmd_body = get("run_command"), get("g129")
        self._hand_cmd_norm = np.full(12, np.nan, np.float32)   # rt/inspire/cmd as sent: normalized 0..1
        hand = get("inspire")
        if hand is not None and hasattr(hand, "register_hand_cmd_callback"):
            hand.register_hand_cmd_callback(self._on_hand_cmd)

        self._q: queue.Queue[str] = queue.Queue()
        self._buf: dict[str, list] | None = None
        self._label = ""
        self._t0_wall = 0.0
        self._sub = None
        self._last_cmd = ("", 0.0)
        self._start_listener()
        print(f"[locomanip] ready: {len(self.hand_names)} hand joints, objects={list(self.objects)}, "
              f"out={self.out_dir}; control topic {RECORD_TOPIC}")

    # ---------------- control ----------------
    def _start_listener(self):
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            self._sub = ChannelSubscriber(RECORD_TOPIC, String_)
            self._sub.Init(lambda m: self._q.put(m.data.strip()), 10)
        except Exception as e:  # sim keeps running; episodes can still be driven by command()
            print(f"[locomanip] DDS control unavailable ({e}); use recorder.command()")

    def _on_hand_cmd(self, norm_q):
        v = np.asarray(norm_q[:12], np.float32)
        self._hand_cmd_norm = np.pad(v, (0, 12 - len(v)), constant_values=np.nan)

    def command(self, text: str):
        self._q.put(text.strip())

    def _handle(self, text: str):
        now = time.time()
        if text == self._last_cmd[0] and now - self._last_cmd[1] < 1.0:   # cli.py repeats each command
            return
        self._last_cmd = (text, now)
        word, _, arg = text.partition(" ")
        if word == "start":
            if self._buf is not None:
                self._save()
            self._label = (arg.strip() or "episode").replace(" ", "_").replace("/", "_")
            self._buf = {k: [] for k in ("t_sim", "t_wall", "root_pos", "root_quat_wxyz", "root_lin_vel", "root_ang_vel",
                                         "q", "dq", "hand_q", "cmd_vel", "cmd_arm_q", "cmd_hand_q")}
            for o in self.objects:
                self._buf[f"obj_{o}"] = []
            self._t0_wall = time.time()
            print(f"[locomanip] REC start '{self._label}'")
        elif word == "stop":
            if self._buf is not None:
                self._save()
            self._buf = None
        elif word == "discard":
            print(f"[locomanip] REC discarded '{self._label}' ({len(self._buf['t_sim']) if self._buf else 0} steps)")
            self._buf = None
        else:
            print(f"[locomanip] unknown command '{text}' (start <label> | stop | discard)")

    # ---------------- per step ----------------
    def step(self):
        while not self._q.empty():
            self._handle(self._q.get_nowait())
        if self._buf is None:
            return
        d, b = self.robot.data, self._buf
        b["t_sim"].append(float(self.env.sim.current_time))
        b["t_wall"].append(time.time() - self._t0_wall)
        b["root_pos"].append(_np(d.root_pos_w[0]))
        b["root_quat_wxyz"].append(_np(d.root_quat_w[0]))
        b["root_lin_vel"].append(_np(d.root_lin_vel_w[0]))
        b["root_ang_vel"].append(_np(d.root_ang_vel_w[0]))
        q, dq = _np(d.joint_pos[0]), _np(d.joint_vel[0])
        b["q"].append(q[self.body_idx])
        b["dq"].append(dq[self.body_idx])
        b["hand_q"].append(q[self.hand_idx])
        b["cmd_vel"].append(_parse_run_command(self.cmd_run.get_run_command() if self.cmd_run else None))
        arm = np.full(14, np.nan, np.float32)
        lc = self.cmd_body.get_robot_command() if self.cmd_body else None
        if lc and "motor_cmd" in lc and len(lc["motor_cmd"]["positions"]) >= 29:
            arm = np.asarray(lc["motor_cmd"]["positions"][ARM_LOWCMD_SLICE], np.float32)
        b["cmd_arm_q"].append(arm)
        b["cmd_hand_q"].append(self._hand_cmd_norm.copy())
        for o, obj in self.objects.items():
            b[f"obj_{o}"].append(np.concatenate([_np(obj.data.root_pos_w[0]), _np(obj.data.root_quat_w[0])]))

    def close(self):
        if self._buf is not None:
            self._save()
            self._buf = None

    # ---------------- save ----------------
    def _save(self):
        b = self._buf
        n = len(b["t_sim"])
        if n < 10:
            print(f"[locomanip] '{self._label}' too short ({n} steps), not saved")
            return
        arr = {k: np.asarray(v, np.float32 if k != "t_sim" else np.float64) for k, v in b.items()}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep_dir = os.path.join(self.out_dir, f"{stamp}_{self._label}")
        os.makedirs(ep_dir, exist_ok=True)
        np.savez_compressed(os.path.join(ep_dir, "episode.npz"), **arr,
                            body_joint_names=np.array(BODY_JOINTS_MJ), hand_joint_names=np.array(self.hand_names))
        t = arr["t_sim"] - arr["t_sim"][0]
        csv_path = os.path.join(ep_dir, "motion_50hz.csv")
        frames, offset = write_bones_csv(csv_path, t, arr["root_pos"], arr["root_quat_wxyz"], arr["q"], self.ground_z)
        meta = {"task": self.task, "label": self._label, "steps": n, "duration_sim_s": float(t[-1]),
                "duration_wall_s": float(arr["t_wall"][-1]), "csv_fps": CSV_FPS, "csv_frames": frames,
                "root_xy_offset_m": offset, "ground_z_m": self.ground_z,
                "body_joint_names_mj": BODY_JOINTS_MJ, "hand_joint_names": self.hand_names,
                "objects": list(self.objects), "source": "isaac-sim ground truth"}
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        dist = float(np.linalg.norm(arr["root_pos"][-1, :2] - arr["root_pos"][0, :2]))
        print(f"[locomanip] REC saved '{self._label}': {t[-1]:.1f} s sim, {frames} frames @{CSV_FPS} Hz, "
              f"pelvis moved {dist:.2f} m -> {ep_dir}")


def resample(t, root_pos, root_quat_wxyz, q, fps=CSV_FPS):
    """Uniform resample on sim time: linear for positions/joints, slerp for the pelvis rotation."""
    keep = np.concatenate([[True], np.diff(t) > 1e-6])   # drop repeated sim timestamps
    t, root_pos, root_quat_wxyz, q = t[keep], root_pos[keep], root_quat_wxyz[keep], q[keep]
    tt = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    pos = np.stack([np.interp(tt, t, root_pos[:, i]) for i in range(3)], 1)
    qq = np.stack([np.interp(tt, t, q[:, i]) for i in range(q.shape[1])], 1)
    rot = Slerp(t, Rotation.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]]))(tt)
    return tt, pos, rot, qq


def write_bones_csv(path, t, root_pos, root_quat_wxyz, q_mj, ground_z=0.0, fps=CSV_FPS):
    """Bones-SEED flat CSV: Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg, scipy 'xyz'],
    29 *_dof [deg] in MuJoCo order. Pelvis x/y start at the origin; z is height above ground."""
    tt, pos, rot, qq = resample(t, root_pos, root_quat_wxyz, q_mj, fps)
    offset = [float(pos[0, 0]), float(pos[0, 1])]
    pos = pos - np.array([offset[0], offset[1], ground_z])
    eul = rot.as_euler("xyz", degrees=True)   # inverse of the converter's from_euler("xyz")
    header = (["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"]
              + [f"{j}_dof" for j in BODY_JOINTS_MJ])
    data = np.column_stack([np.arange(len(tt)), pos * 100.0, eul, np.rad2deg(qq)])
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="",
               fmt=["%d"] + ["%.6f"] * (data.shape[1] - 1))
    return len(tt), offset
