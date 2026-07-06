"""Teleop trajectory slots 1–5 ↔ inspire pick-place objects."""

from __future__ import annotations

from dataclasses import dataclass

from .trajectory_io import SLOTS, traj_path

# hand_grip_boost: None → no boost; set per slot (e.g. cardsdeck).


@dataclass(frozen=True)
class TrajectorySlot:
    slot: int
    object_key: str
    label: str
    sim_env: str  # arg to scripts/run_inspire_teleop_env.sh
    freeze_left_arm: bool = False
    left_side_freeze_s: float = 0.0
    left_side_join_ramp_s: float = 2.0
    playback_offset_s: float = 0.0
    hand_grip_boost: float | None = None


TRAJECTORY_SLOTS: dict[int, TrajectorySlot] = {
    1: TrajectorySlot(1, "coin", "coin", "coin"),
    2: TrajectorySlot(2, "stick", "stick", "stick"),
    3: TrajectorySlot(3, "tennisball", "tennisball", "tennisball"),
    4: TrajectorySlot(
        4,
        "cardsdeck",
        "cardsdeck",
        "cardsdeck",
        left_side_freeze_s=5.0,
        playback_offset_s=5.0,
        hand_grip_boost=0.40,
    ),
    5: TrajectorySlot(5, "container", "container", "container", freeze_left_arm=True),
}

# External TCP uint32 LE command → internal recording slot (traj_N.npz).
TCP_TRAJECTORY_CMD: dict[int, int] = {
    1: 5,  # container
    2: 2,  # stick
    4: 3,  # tennisball
    6: 4,  # cardsdeck
}


def slot_from_tcp_cmd(cmd: int) -> int:
    if cmd not in TCP_TRAJECTORY_CMD:
        valid = ", ".join(str(k) for k in sorted(TCP_TRAJECTORY_CMD))
        raise ValueError(f"unknown TCP cmd {cmd}; valid: {valid}")
    return TCP_TRAJECTORY_CMD[cmd]


def tcp_cmd_help() -> str:
    parts = []
    for tcp, slot in sorted(TCP_TRAJECTORY_CMD.items()):
        info = TRAJECTORY_SLOTS[slot]
        parts.append(f"tcp {tcp} → slot {slot} ({info.label} / {info.sim_env})")
    return "; ".join(parts)


def get_slot(slot: int) -> TrajectorySlot:
    if slot not in TRAJECTORY_SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}, got {slot}")
    return TRAJECTORY_SLOTS[slot]


def slot_has_recording(slot: int) -> bool:
    return traj_path(slot).exists()
