"""Load/save teleop trajectories (dual arm + Inspire hands) in numbered slots 1-5."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .g1_constants import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"
SLOTS = (1, 2, 3, 4, 5)


def traj_path(slot: int) -> Path:
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}, got {slot}")
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    return TRAJ_DIR / f"traj_{slot}.npz"


def save_trajectory(
    slot: int,
    *,
    left_arm_q: np.ndarray,
    right_arm_q: np.ndarray,
    hand_q: np.ndarray,
    body_cmd: np.ndarray | None = None,
    body_q: np.ndarray | None = None,
    hz: float = 30.0,
    label: str = "",
    record_source: str = "",
) -> Path:
    path = traj_path(slot)
    payload = {
        "slot": np.int32(slot),
        "hz": np.float32(hz),
        "label": np.array(label or f"trajectory_{slot}"),
        "left_arm_q": np.asarray(left_arm_q, dtype=np.float32),
        "right_arm_q": np.asarray(right_arm_q, dtype=np.float32),
        "hand_q": np.asarray(hand_q, dtype=np.float32),
    }
    if body_cmd is not None:
        payload["body_cmd"] = np.asarray(body_cmd, dtype=np.float32)
    if body_q is not None:
        payload["body_q"] = np.asarray(body_q, dtype=np.float32)
    if record_source:
        payload["record_source"] = np.array(record_source)
    np.savez_compressed(path, **payload)
    return path


def load_trajectory(slot: int) -> dict:
    path = traj_path(slot)
    if not path.exists():
        raise FileNotFoundError(f"No trajectory in slot {slot}: {path}")
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    label = out.get("label")
    if isinstance(label, np.ndarray):
        out["label"] = str(label)
    src = out.get("record_source")
    if isinstance(src, np.ndarray):
        out["record_source"] = str(src)
    out["hz"] = float(out.get("hz", 30.0))
    out["slot"] = int(out.get("slot", slot))
    return out


def list_trajectories() -> list[dict]:
    rows = []
    for slot in SLOTS:
        path = traj_path(slot)
        if not path.exists():
            rows.append({"slot": slot, "path": path, "frames": 0, "hz": 0.0, "label": "(empty)", "duration_s": 0.0})
            continue
        traj = load_trajectory(slot)
        n = int(traj["left_arm_q"].shape[0])
        hz = float(traj["hz"])
        rows.append({
            "slot": slot,
            "path": path,
            "frames": n,
            "hz": hz,
            "label": traj.get("label", f"trajectory_{slot}"),
            "duration_s": n / hz if hz > 0 else 0.0,
        })
    return rows


def trim_trajectory(
    slot: int,
    *,
    tail_s: float = 0.0,
    head_s: float = 0.0,
) -> Path:
    """Drop ``head_s`` / ``tail_s`` seconds from the start/end of a saved trajectory."""
    if tail_s < 0 or head_s < 0:
        raise ValueError("head_s and tail_s must be >= 0")
    traj = load_trajectory(slot)
    hz = float(traj["hz"])
    n = int(traj["left_arm_q"].shape[0])
    head_f = int(round(head_s * hz))
    tail_f = int(round(tail_s * hz))
    end = n - tail_f
    if head_f >= end:
        raise ValueError(f"trim removes all frames (head={head_f}, end={end}, n={n})")
    sl = slice(head_f, end)
    kwargs = {
        "left_arm_q": traj["left_arm_q"][sl],
        "right_arm_q": traj["right_arm_q"][sl],
        "hand_q": traj["hand_q"][sl],
        "hz": hz,
        "label": traj.get("label", f"trajectory_{slot}"),
        "record_source": traj.get("record_source", ""),
    }
    if "body_cmd" in traj:
        kwargs["body_cmd"] = traj["body_cmd"][sl]
    if "body_q" in traj:
        kwargs["body_q"] = traj["body_q"][sl]
    return save_trajectory(slot, **kwargs)
