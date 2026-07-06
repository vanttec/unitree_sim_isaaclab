"""Record dual-arm + Inspire trajectories from DDS while teleoperating."""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from .dds_env import configure_local_dds, dds_interface
from .g1_constants import G1_NUM_MOTOR, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from .trajectory_io import save_trajectory


class TrajectoryRecorder:
    """Buffers rt/lowcmd + rt/inspire/cmd (what the sim actually follows)."""

    def __init__(self, domain_channel: int = 1, network_interface: str | None = None, local_dds: bool = True):
        self._lock = threading.Lock()
        self._low_state: LowState_ | None = None
        self._low_cmd: LowCmd_ | None = None
        self._hand_q: list[float] | None = None
        self._hand_cmd_q: list[float] | None = None
        self._recording = False
        self._body_cmd_buf: list[np.ndarray] = []
        self._body_state_buf: list[np.ndarray] = []
        self._left_buf: list[np.ndarray] = []
        self._right_buf: list[np.ndarray] = []
        self._hand_buf: list[np.ndarray] = []
        self._hz = 30.0

        if local_dds:
            configure_local_dds()
        iface = dds_interface(network_interface)
        ChannelFactoryInitialize(domain_channel, iface)

        self._sub_low = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub_low.Init(self._on_low_state, 10)
        self._sub_cmd_body = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._sub_cmd_body.Init(self._on_low_cmd, 10)
        self._sub_hand = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self._sub_hand.Init(self._on_hand_state, 10)
        self._sub_cmd = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
        self._sub_cmd.Init(self._on_hand_cmd, 10)

    def _on_low_cmd(self, msg: LowCmd_) -> None:
        with self._lock:
            self._low_cmd = msg

    def _on_hand_cmd(self, msg: MotorCmds_) -> None:
        with self._lock:
            self._hand_cmd_q = [float(c.q) for c in msg.cmds[:12]]

    def _on_low_state(self, msg: LowState_) -> None:
        with self._lock:
            self._low_state = msg

    def _on_hand_state(self, msg: MotorStates_) -> None:
        with self._lock:
            self._hand_q = [float(s.q) for s in msg.states[:12]]

    def wait_for_state(self, timeout_s: float = 30.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if (
                    self._low_cmd is not None
                    and (self._hand_cmd_q is not None or self._hand_q is not None)
                ):
                    return
            time.sleep(0.05)
        raise TimeoutError("No rt/lowcmd or rt/inspire/cmd. Is sim + teleop running with [r]?")

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        with self._lock:
            if self._low_cmd is None:
                return None
            hand_src = self._hand_cmd_q if self._hand_cmd_q is not None else self._hand_q
            if hand_src is None:
                return None
            body_cmd = np.array(
                [self._low_cmd.motor_cmd[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64
            )
            if self._low_state is not None:
                body_state = np.array(
                    [self._low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64
                )
            else:
                body_state = body_cmd.copy()
            left_q = np.array([body_cmd[i] for i in LEFT_ARM_JOINTS], dtype=np.float64)
            right_q = np.array([body_cmd[i] for i in RIGHT_ARM_JOINTS], dtype=np.float64)
            hand_q = np.array(hand_src[:12], dtype=np.float64)
        return body_cmd, body_state, left_q, right_q, hand_q

    def record_until(self, stop: Callable[[], bool], hz: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._hz = hz
        dt = 1.0 / hz
        self._body_cmd_buf.clear()
        self._body_state_buf.clear()
        self._left_buf.clear()
        self._right_buf.clear()
        self._hand_buf.clear()
        self._recording = True
        try:
            while not stop():
                t0 = time.time()
                snap = self._snapshot()
                if snap is not None:
                    body_cmd, body_state, left_q, right_q, hand_q = snap
                    self._body_cmd_buf.append(body_cmd)
                    self._body_state_buf.append(body_state)
                    self._left_buf.append(left_q)
                    self._right_buf.append(right_q)
                    self._hand_buf.append(hand_q)
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        finally:
            self._recording = False

        if not self._body_cmd_buf:
            raise RuntimeError("No frames recorded")

        return (
            np.stack(self._body_cmd_buf),
            np.stack(self._left_buf),
            np.stack(self._right_buf),
            np.stack(self._hand_buf),
        )

    def save_slot(self, slot: int, label: str = "") -> str:
        if not self._body_cmd_buf:
            raise RuntimeError("Nothing recorded yet")
        path = save_trajectory(
            slot,
            left_arm_q=np.stack(self._left_buf),
            right_arm_q=np.stack(self._right_buf),
            hand_q=np.stack(self._hand_buf),
            body_cmd=np.stack(self._body_cmd_buf),
            body_q=np.stack(self._body_state_buf),
            hz=self._hz,
            label=label or f"trajectory_{slot}",
            record_source="lowcmd+inspire_cmd",
        )
        return str(path)
