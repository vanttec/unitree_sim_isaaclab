"""Replay a saved dual-arm + Inspire trajectory over DDS."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from .dds_client import UnitreeDdsClient
from .g1_constants import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from .trajectory_io import load_trajectory

DEFAULT_HAND_GRIP_BOOST = 0.0


def _boost_hand_grip(hand: np.ndarray, boost: float) -> np.ndarray:
    """Pull normalized Inspire cmds toward closed (q=0) when already grasping."""
    if boost <= 0.0:
        return hand
    out = np.array(hand, dtype=np.float64, copy=True)
    for i in range(12):
        if out[i] < 0.75:
            out[i] = max(0.0, out[i] * (1.0 - boost))
    return out


@dataclass(frozen=True)
class LeftSideHold:
    """Left arm + left Inspire hand locked to live sim pose (slot 5 only)."""

    arm_q: np.ndarray
    hand_q: np.ndarray


@dataclass(frozen=True)
class RightSideHold:
    """Right arm + right Inspire hand locked to live sim pose."""

    arm_q: np.ndarray
    hand_q: np.ndarray


@dataclass(frozen=True)
class OffsetReplayPlan:
    """Slot 4: freeze at sim pose, ramp into recording at playback_offset_s, then play."""

    base_body: np.ndarray
    base_hand: np.ndarray
    left_sim: LeftSideHold
    right_sim: RightSideHold
    freeze_s: float
    playback_offset_s: float
    join_ramp_s: float = 2.0


def _capture_left_side_hold(client: UnitreeDdsClient) -> LeftSideHold:
    body = client.current_joint_positions()
    arm_q = np.array([body[i] for i in LEFT_ARM_JOINTS], dtype=np.float64)
    hand_q = client.current_hand_positions()
    return LeftSideHold(arm_q=arm_q, hand_q=np.array(hand_q[6:12], dtype=np.float64))


def _capture_right_side_hold(client: UnitreeDdsClient) -> RightSideHold:
    body = client.current_joint_positions()
    hand_q = client.current_hand_positions()
    return RightSideHold(
        arm_q=np.array([body[i] for i in RIGHT_ARM_JOINTS], dtype=np.float64),
        hand_q=np.array(hand_q[:6], dtype=np.float64),
    )


def _apply_left_side_hold(body: np.ndarray, hand: np.ndarray, hold: LeftSideHold) -> None:
    for j, idx in enumerate(LEFT_ARM_JOINTS):
        body[idx] = float(hold.arm_q[j])
    hand[6:12] = hold.hand_q


def publish_reset_category(category: int, channel: int = 1) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(channel)
    pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    pub.Init()
    pub.Write(String_(data=str(category)))
    print(f"[traj] reset category {category} sent")


class TrajectoryExecutor:
    def __init__(
        self,
        client: UnitreeDdsClient,
        hand_kp: float = 0.0,
        hand_kd: float = 0.0,
        hand_grip_boost: float = DEFAULT_HAND_GRIP_BOOST,
    ):
        self._client = client
        self._hand_kp = hand_kp
        self._hand_kd = hand_kd
        self._hand_grip_boost = hand_grip_boost

    def _publish_frame(self, body: np.ndarray, hand: np.ndarray) -> None:
        hand_cmd = _boost_hand_grip(hand, self._hand_grip_boost)
        self._client.publish_lowcmd(body)
        self._client.publish_inspire(hand_cmd, hand_kp=self._hand_kp, hand_kd=self._hand_kd)

    def _recording_frame(self, traj: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
        body_cmd = traj.get("body_cmd")
        body_q = traj.get("body_q")
        body_src = body_cmd if body_cmd is not None else body_q
        if body_src is not None:
            targets = np.array(body_src[i], dtype=np.float64)
        else:
            targets = self._client.current_joint_positions()
            for j, idx in enumerate(LEFT_ARM_JOINTS):
                targets[idx] = float(traj["left_arm_q"][i, j])
            for j, idx in enumerate(RIGHT_ARM_JOINTS):
                targets[idx] = float(traj["right_arm_q"][i, j])
        hand_q = np.array(traj["hand_q"][i], dtype=np.float64)
        return targets, hand_q

    def _frame_targets(
        self,
        traj: dict,
        i: int,
        *,
        left_hold: LeftSideHold | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets, hand_q = self._recording_frame(traj, i)

        if left_hold is None:
            return targets, hand_q

        out_body = self._client.current_joint_positions()
        for j, idx in enumerate(RIGHT_ARM_JOINTS):
            out_body[idx] = targets[idx]
        try:
            out_hand = self._client.current_hand_positions().copy()
        except RuntimeError:
            out_hand = hand_q.copy()
        out_hand[:6] = hand_q[:6]
        _apply_left_side_hold(out_body, out_hand, left_hold)
        return out_body, out_hand

    def _ramp_to_start(
        self,
        traj: dict,
        ramp_s: float,
        hz: float,
        *,
        left_hold: LeftSideHold | None = None,
        start_frame: int = 0,
    ) -> None:
        dt = 1.0 / hz
        start_body, start_hand = self._recording_frame(traj, start_frame)
        current_body = self._client.current_joint_positions()
        try:
            current_hand = self._client.current_hand_positions()
        except RuntimeError:
            current_hand = start_hand.copy()

        if left_hold is not None:
            right_body_gap = float(
                max(abs(start_body[i] - current_body[i]) for i in RIGHT_ARM_JOINTS)
            )
            right_hand_gap = float(max(abs(start_hand[j] - current_hand[j]) for j in range(6)))
            if right_body_gap < 0.05 and right_hand_gap < 0.05:
                print(
                    f"[traj] Right side already at start pose "
                    f"(arm gap {right_body_gap:.3f}, hand gap {right_hand_gap:.3f}); left locked"
                )
                return
            print(
                f"[traj] Ramping right side → frame {start_frame} over {ramp_s:.1f}s "
                f"(arm gap {right_body_gap:.2f} rad, hand gap {right_hand_gap:.2f}); left locked"
            )
        else:
            body_gap = float(np.max(np.abs(start_body - current_body)))
            hand_gap = float(np.max(np.abs(start_hand - current_hand)))
            if body_gap < 0.05 and hand_gap < 0.05:
                print(f"[traj] Already at start pose (body gap {body_gap:.3f}, hand gap {hand_gap:.3f})")
                return
            label = f"frame {start_frame}" if start_frame else "grabación frame 0"
            print(
                f"[traj] Interpolating sim → {label} over {ramp_s:.1f}s "
                f"(body gap {body_gap:.2f} rad, hand gap {hand_gap:.2f})..."
            )

        steps = max(1, int(ramp_s * hz))
        for step in range(steps):
            alpha = (step + 1) / steps
            body_cmd = current_body.copy()
            hand_cmd = current_hand.copy()
            for idx in RIGHT_ARM_JOINTS:
                body_cmd[idx] = (1.0 - alpha) * current_body[idx] + alpha * start_body[idx]
            for j in range(6):
                hand_cmd[j] = (1.0 - alpha) * current_hand[j] + alpha * start_hand[j]
            if left_hold is not None:
                _apply_left_side_hold(body_cmd, hand_cmd, left_hold)
            else:
                for idx in LEFT_ARM_JOINTS:
                    body_cmd[idx] = (1.0 - alpha) * current_body[idx] + alpha * start_body[idx]
                for j in range(6, 12):
                    hand_cmd[j] = (1.0 - alpha) * current_hand[j] + alpha * start_hand[j]
            self._publish_frame(body_cmd, hand_cmd)
            time.sleep(dt)

    def _play_offset_replay(
        self,
        traj: dict,
        hz: float,
        plan: OffsetReplayPlan,
    ) -> None:
        dt = 1.0 / hz
        n = int(traj["left_arm_q"].shape[0])
        offset_i = min(int(plan.playback_offset_s * hz), max(n - 1, 0))
        offset_body, offset_hand = self._recording_frame(traj, offset_i)

        freeze_steps = max(1, int(plan.freeze_s * hz))
        ramp_steps = max(1, int(plan.join_ramp_s * hz))
        play_steps = max(0, n - offset_i)
        total_steps = freeze_steps + ramp_steps + play_steps

        print(
            f"[traj] Offset replay: freeze {plan.freeze_s:.0f}s at sim pose, "
            f"ramp {plan.join_ramp_s:.0f}s → recording t={plan.playback_offset_s:.0f}s "
            f"(frame {offset_i}), then {play_steps} frames"
        )

        t_play = time.perf_counter()
        for step in range(total_steps):
            if step < freeze_steps:
                body = plan.base_body.copy()
                hand = plan.base_hand.copy()
            elif step < freeze_steps + ramp_steps:
                alpha = (step - freeze_steps + 1) / ramp_steps
                body = plan.base_body.copy()
                hand = plan.base_hand.copy()
                for j, idx in enumerate(LEFT_ARM_JOINTS):
                    body[idx] = (1.0 - alpha) * plan.left_sim.arm_q[j] + alpha * offset_body[idx]
                for j, idx in enumerate(RIGHT_ARM_JOINTS):
                    body[idx] = (1.0 - alpha) * plan.right_sim.arm_q[j] + alpha * offset_body[idx]
                for j in range(6):
                    hand[j] = (1.0 - alpha) * plan.right_sim.hand_q[j] + alpha * offset_hand[j]
                for j in range(6):
                    hand[6 + j] = (1.0 - alpha) * plan.left_sim.hand_q[j] + alpha * offset_hand[6 + j]
            else:
                play_i = min(offset_i + (step - freeze_steps - ramp_steps), n - 1)
                body, hand = self._recording_frame(traj, play_i)

            self._publish_frame(body, hand)
            t_next = t_play + (step + 1) * dt
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _capture_offset_plan(self, client: UnitreeDdsClient, slot_info) -> OffsetReplayPlan:
        base_body = client.current_joint_positions()
        base_hand = client.current_hand_positions().copy()
        return OffsetReplayPlan(
            base_body=base_body,
            base_hand=base_hand,
            left_sim=_capture_left_side_hold(client),
            right_sim=_capture_right_side_hold(client),
            freeze_s=slot_info.left_side_freeze_s,
            playback_offset_s=slot_info.playback_offset_s,
            join_ramp_s=slot_info.left_side_join_ramp_s,
        )

    def run(
        self,
        slot: int,
        *,
        loop: bool = False,
        ramp_s: float = 3.0,
        reset_object: bool = False,
        reset_channel: int = 1,
        freeze_left_arm: bool = False,
    ) -> None:
        from .trajectory_registry import get_slot

        slot_info = get_slot(slot)
        saved_grip_boost = self._hand_grip_boost
        if slot_info.hand_grip_boost is not None:
            self._hand_grip_boost = slot_info.hand_grip_boost
        traj = load_trajectory(slot)
        hz = float(traj["hz"])
        dt = 1.0 / hz
        label = traj.get("label", f"trajectory_{slot}")
        n = int(traj["left_arm_q"].shape[0])

        print(f"[traj] Playing slot {slot} ({label}) — {n} frames @ {hz:.1f} Hz ({n/hz:.1f}s)")
        src = traj.get("record_source", "")
        if traj.get("body_cmd") is None:
            print("[traj] ⚠ Grabación vieja (solo lowstate). Re-graba para que coincida con teleop.")
        elif src:
            print(f"[traj] Source: {src}")
        print("[traj] IMPORTANTE: para el teleop antes de reproducir (Ctrl+C en teleop_hand_and_arm.py)")
        if slot_info.hand_grip_boost is not None:
            print(f"[traj] hand grip boost: {self._hand_grip_boost:.2f} (slot {slot})")

        if reset_object:
            publish_reset_category(1, channel=reset_channel)
            time.sleep(1.5)

        use_offset = slot_info.playback_offset_s > 0.0
        left_hold = _capture_left_side_hold(self._client) if freeze_left_arm else None
        if left_hold is not None:
            print("[traj] Left arm + left hand locked to current sim pose (only right side replays)")

        if not use_offset:
            self._ramp_to_start(traj, ramp_s=ramp_s, hz=hz, left_hold=left_hold)

        try:
            while True:
                if use_offset:
                    self._play_offset_replay(traj, hz, self._capture_offset_plan(self._client, slot_info))
                else:
                    t_play = time.perf_counter()
                    for i in range(n):
                        targets, hand_q = self._frame_targets(traj, i, left_hold=left_hold)
                        self._publish_frame(targets, hand_q)
                        t_next = t_play + (i + 1) * dt
                        time.sleep(max(0.0, t_next - time.perf_counter()))

                if not loop:
                    break
                print(f"[traj] Looping slot {slot}...")
                if not use_offset:
                    self._ramp_to_start(traj, ramp_s=ramp_s, hz=hz, left_hold=left_hold)
        except KeyboardInterrupt:
            print("[traj] Interrupted")
        finally:
            self._hand_grip_boost = saved_grip_boost

        print(f"[traj] Done slot {slot}")


def play_trajectory_slot(
    executor: TrajectoryExecutor,
    slot: int,
    *,
    ramp_s: float = 2.0,
    reset_object: bool = False,
    reset_channel: int = 1,
    freeze_left_arm: bool | None = None,
) -> None:
    from .trajectory_registry import get_slot

    info = get_slot(slot)
    freeze = info.freeze_left_arm if freeze_left_arm is None else freeze_left_arm
    print(f"[traj] slot {slot} ({info.label}) — sim env should be: {info.sim_env}")
    executor.run(
        slot,
        ramp_s=ramp_s,
        reset_object=reset_object,
        reset_channel=reset_channel,
        freeze_left_arm=freeze,
    )
