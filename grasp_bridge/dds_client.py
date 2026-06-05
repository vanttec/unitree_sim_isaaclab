"""Unitree SDK2 DDS client for G1 body + Inspire hands (sim or real robot)."""

from __future__ import annotations

import os
import threading
import time
from typing import Iterable

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from .dds_env import configure_local_dds, dds_interface
from .g1_constants import ARM_JOINTS, G1_NUM_MOTOR, KD, KP


class UnitreeDdsClient:
    """Publish rt/lowcmd + rt/inspire/cmd; subscribe rt/lowstate."""

    def __init__(
        self,
        domain_channel: int = 1,
        network_interface: str | None = None,
        local_dds: bool = True,
    ):
        self._crc = CRC()
        self._low_state: LowState_ | None = None
        self._state_lock = threading.Lock()
        self._mode_machine = 0
        self._state_count = 0

        if local_dds:
            configure_local_dds()

        iface = dds_interface(network_interface)
        if iface:
            print(f"[dds] ChannelFactoryInitialize({domain_channel}, '{iface}')")
        else:
            print(f"[dds] ChannelFactoryInitialize({domain_channel}, autodetermine)")
        ChannelFactoryInitialize(domain_channel, iface)

        self._low_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._low_pub.Init()

        self._hand_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._hand_pub.Init()

        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_) -> None:
        with self._state_lock:
            self._low_state = msg
            self._mode_machine = int(msg.mode_machine)
            self._state_count += 1

    def wait_for_state(self, timeout_s: float = 30.0) -> None:
        deadline = time.time() + timeout_s
        last_log = 0.0
        while time.time() < deadline:
            with self._state_lock:
                if self._low_state is not None:
                    print(f"[dds] rt/lowstate OK ({self._state_count} msgs)")
                    return
            now = time.time()
            if now - last_log > 5.0:
                iface = os.environ.get("UNITREE_DDS_INTERFACE", "autodetermine")
                print(f"[dds] waiting for rt/lowstate (iface={iface})...")
                last_log = now
            time.sleep(0.05)

        iface = os.environ.get("UNITREE_DDS_INTERFACE", "autodetermine")
        raise TimeoutError(
            "No rt/lowstate received.\n"
            "Checklist:\n"
            "  1) sim_main.py running AND past 'start controller success'\n"
            "  2) In BOTH terminals BEFORE starting sim:\n"
            "       source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh\n"
            f"  3) Same interface in both (current: {iface})\n"
            "  4) Restart sim after sourcing setup script\n"
            "  5) Test: python -m grasp_bridge.dds_ping"
        )

    def current_joint_positions(self) -> np.ndarray:
        with self._state_lock:
            if self._low_state is None:
                raise RuntimeError("lowstate not available")
            return np.array([self._low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)

    def publish_lowcmd(self, joint_targets: Iterable[float], kp: Iterable[float] | None = None, kd: Iterable[float] | None = None) -> None:
        targets = list(joint_targets)
        if len(targets) < G1_NUM_MOTOR:
            raise ValueError(f"expected {G1_NUM_MOTOR} joint targets, got {len(targets)}")

        kp_vals = list(kp) if kp is not None else list(KP)
        kd_vals = list(kd) if kd is not None else list(KD)

        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = 0
        cmd.mode_machine = self._mode_machine

        for i in range(G1_NUM_MOTOR):
            mc = cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(targets[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(kp_vals[i])
            mc.kd = float(kd_vals[i])

        cmd.crc = self._crc.Crc(cmd)
        self._low_pub.Write(cmd)

    def publish_inspire(self, hand_q: Iterable[float], hand_kp: float = 1.0, hand_kd: float = 0.05) -> None:
        values = list(hand_q)
        if len(values) != 12:
            raise ValueError(f"expected 12 inspire motor commands, got {len(values)}")

        msg = MotorCmds_(cmds=[])
        for q in values:
            mc = unitree_go_msg_dds__MotorCmd_()
            mc.q = float(np.clip(q, 0.0, 1.0))
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = hand_kp
            mc.kd = hand_kd
            msg.cmds.append(mc)

        self._hand_pub.Write(msg)
