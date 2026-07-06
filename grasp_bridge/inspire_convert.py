"""Convert Inspire hand joint angles (Isaac Sim / MJCF radians) to DDS normalized [0, 1]."""

from __future__ import annotations

import numpy as np

# Same ranges as dds/inspire_dds.py (DDS motor index 0-11).
_DDS_RANGES: dict[int, tuple[float, float]] = {
    0: (0.0, 1.7),
    1: (0.0, 1.7),
    2: (0.0, 1.7),
    3: (0.0, 1.7),
    4: (0.0, 0.5),
    5: (-0.1, 1.3),
    6: (0.0, 1.7),
    7: (0.0, 1.7),
    8: (0.0, 1.7),
    9: (0.0, 1.7),
    10: (0.0, 0.5),
    11: (-0.1, 1.3),
}

# DDS index -> Isaac joint name (proximal / yaw only).
DDS_INSPIRE_JOINT_NAMES: tuple[str, ...] = (
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
)

_LEFT_DDS_INDICES = (6, 7, 8, 9, 10, 11)
_RIGHT_DDS_INDICES = (0, 1, 2, 3, 4, 5)
_HAND_KWARGS = ("pinky", "ring", "middle", "index", "thumb_p", "thumb_y")


def rad_to_dds_norm(rad: float, dds_index: int) -> float:
    """Isaac joint rad → DDS norm in [0, 1] (0=closed, 1=open)."""
    lo, hi = _DDS_RANGES[dds_index]
    return float(np.clip((hi - rad) / (hi - lo), 0.0, 1.0))


def dds_norm_to_rad(norm: float, dds_index: int) -> float:
    """Inverse of rad_to_dds_norm (matches inspire_dds.denormalize)."""
    lo, hi = _DDS_RANGES[dds_index]
    n = float(np.clip(norm, 0.0, 1.0))
    return (1.0 - n) * (hi - lo) + lo


def inspire_radians_by_name(joint_rad: dict[str, float]) -> tuple[float, ...]:
    """12 DDS-normalized values from a name -> rad map (missing joints = open)."""
    out = [0.0] * 12
    for i, name in enumerate(DDS_INSPIRE_JOINT_NAMES):
        if name in joint_rad:
            out[i] = rad_to_dds_norm(float(joint_rad[name]), i)
    return tuple(out)


def hand_call_from_dds(hand_q: tuple[float, ...], side: str = "left") -> str:
    """Format _hand_left(...) or _hand_right(...) for grasp_library.py."""
    indices = _LEFT_DDS_INDICES if side == "left" else _RIGHT_DDS_INDICES
    if all(hand_q[i] > 0.95 for i in indices):
        return "INSPIRE_OPEN"
    prefix = "l" if side == "left" else "r"
    parts = []
    for kw, idx in zip(_HAND_KWARGS, indices):
        val = round(hand_q[idx], 3)
        if val > 1e-3 and val < 0.95:  # skip open fingers (DDS ~1.0 = open)
            parts.append(f"{kw}={val}")
    fn = f"_hand_{side}"
    if not parts:
        return "INSPIRE_OPEN"
    return f"{fn}({', '.join(parts)})"
