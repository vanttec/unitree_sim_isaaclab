"""Convert g1_replay trajectory_g1.npz to a grasp_library GraspSequence snippet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Driven-joint MJCF ranges (max value = fully closed), from
# do-as-i-do/deployment/g1_replay/g1_29dof_inspire_hand.xml.
_FINGER_MAX = {
    "thumb_proximal_yaw": 1.308,
    "thumb_proximal_pitch": 0.6,
    "index_proximal": 1.47,
    "middle_proximal": 1.47,
    "ring_proximal": 1.47,
    "pinky_proximal": 1.47,
}

# finger_qpos column order (see g1_replay/replay_retarget.py FINGER_SUFFIXES).
_FINGER_COLS = [
    "thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_intermediate", "thumb_distal",
    "index_proximal", "index_intermediate",
    "middle_proximal", "middle_intermediate",
    "ring_proximal", "ring_intermediate",
    "pinky_proximal", "pinky_intermediate",
]

# DDS hand_q order: pinky, ring, middle, index, thumb_pitch, thumb_yaw.
_DDS_ORDER = ["pinky_proximal", "ring_proximal", "middle_proximal", "index_proximal",
              "thumb_proximal_pitch", "thumb_proximal_yaw"]


def finger_qpos_to_dds(finger_qpos_row: np.ndarray) -> tuple[float, ...]:
    col = {name: float(finger_qpos_row[i]) for i, name in enumerate(_FINGER_COLS)}
    return tuple(
        float(np.clip(col[name] / _FINGER_MAX[name], 0.0, 1.0))
        for name in _DDS_ORDER
    )


def _parse_keyframes(spec: str) -> list[tuple[str, int, float]]:
    out = []
    for part in spec.split(","):
        name, rest = part.split("=")
        frame_str, dur_str = rest.split(":")
        out.append((name.strip(), int(frame_str), float(dur_str)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--grasp-id", type=int, required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--keyframes", type=str, default="approach=0:5.0,close=-1:3.0,lift=-1:3.0",
                    help='"name=frame_idx:duration_s,..." — frame_idx=-1 means last frame')
    args = p.parse_args()

    with np.load(args.traj) as npz:
        arm_qpos = np.asarray(npz["arm_qpos"])
        finger_qpos = np.asarray(npz["finger_qpos"])
        side = str(npz["side"])

    n_frames = arm_qpos.shape[0]
    kfs = _parse_keyframes(args.keyframes)

    print(f"# side={side}  n_frames={n_frames}")
    print(f"# Generated from {args.traj}\n")

    hand_prefix = "l_" if side == "left" else "r_"

    print(f"def _sequence_{args.grasp_id}() -> GraspSequence:")
    print(f"    return GraspSequence(")
    print(f"        grasp_id={args.grasp_id},")
    print(f'        label="{args.label}",')
    print(f'        active_side="{side}",')
    print(f"        keyframes=(")
    for name, frame_idx, dur in kfs:
        f = frame_idx if frame_idx >= 0 else n_frames - 1
        if not (0 <= f < n_frames):
            raise ValueError(f"keyframe '{name}' frame {f} out of range [0, {n_frames})")
        arm = tuple(round(float(v), 4) for v in arm_qpos[f])
        dds = finger_qpos_to_dds(finger_qpos[f])
        hand_kwargs = {f"{hand_prefix}{n.replace('_proximal', '').replace('proximal_', '')}": round(v, 3)
                       for n, v in zip(["pinky", "ring", "middle", "index", "thumb_p", "thumb_y"], dds)
                       if abs(v) > 1e-4}
        hand_call = "_hand_" + side + "(" + ", ".join(f"{k.split('_', 1)[1]}={v}" for k, v in hand_kwargs.items()) + ")"
        # left_arm / right_arm — fill the active side with the solved pose,
        # the inactive side is a don't-care (executor never reads it).
        left_arm = arm if side == "left" else (0.0,) * 7
        right_arm = arm if side == "right" else (0.0,) * 7
        print(f"            GraspKeyframe(\"{name}\", {dur}, {left_arm}, {right_arm}, {hand_call}),")
    print(f"        ),")
    print(f"    )")
    print()
    print(f"# Add to GRASPS dict: {args.grasp_id}: _sequence_{args.grasp_id}(),")


if __name__ == "__main__":
    main()
