"""Convert xr_teleoperate episode data.json into a compact trajectory .npz."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .trajectory_io import save_trajectory


def convert_episode_json(episode_json: Path, slot: int, hz: float = 30.0, label: str = "") -> Path:
    with open(episode_json, encoding="utf-8") as f:
        doc = json.load(f)

    frames = doc["data"]
    left_arm, right_arm, hand = [], [], []

    for item in frames:
        act = item["actions"]
        left_arm.append(act["left_arm"]["qpos"])
        right_arm.append(act["right_arm"]["qpos"])
        l_ee = act["left_ee"]["qpos"]
        r_ee = act["right_ee"]["qpos"]
        # teleop inspire: left_ee 6 + right_ee 6; DDS order is right[0:6] then left[6:12]
        if len(l_ee) == 6 and len(r_ee) == 6:
            hand.append(list(r_ee) + list(l_ee))
        elif len(l_ee) == 7 and len(r_ee) == 7:
            hand.append(list(r_ee) + list(l_ee))
        else:
            hand.append(list(r_ee) + list(l_ee))

    info_hz = doc.get("info", {}).get("image", {}).get("fps")
    if info_hz:
        hz = float(info_hz)

    return save_trajectory(
        slot,
        left_arm_q=np.asarray(left_arm, dtype=np.float32),
        right_arm_q=np.asarray(right_arm, dtype=np.float32),
        hand_q=np.asarray(hand, dtype=np.float32),
        hz=hz,
        label=label or f"from_{episode_json.parent.name}",
    )
