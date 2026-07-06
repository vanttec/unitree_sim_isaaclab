#!/usr/bin/env python3
"""Generate grasp_library.py snippets from Isaac Sim poses (YAML or live DDS)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .g1_constants import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from .inspire_convert import (
    DDS_INSPIRE_JOINT_NAMES,
    hand_call_from_dds,
    inspire_radians_by_name,
)

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_SESSION_DEFAULT = Path("grasp_bridge/poses/.capture_session.json")


def _round_tuple(vals: tuple[float, ...], n: int = 4) -> tuple[float, ...]:
    return tuple(round(float(v), n) for v in vals)


def _parse_csv7(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 7:
        raise ValueError(f"expected 7 comma-separated values, got {len(parts)}")
    return tuple(float(p) for p in parts)


def _arm_from_named(d: dict[str, Any], names: tuple[str, ...]) -> tuple[float, ...]:
    missing = [n for n in names if n not in d]
    if missing:
        raise ValueError(f"missing arm joints: {missing}")
    return tuple(float(d[n]) for n in names)


def _hand_from_yaml_block(block: dict[str, Any] | None, side: str) -> tuple[float, ...]:
    if not block:
        return (0.0,) * 12
    if "dds_normalized" in block:
        vals = block["dds_normalized"]
        if len(vals) != 12:
            raise ValueError("dds_normalized must have 12 values")
        return tuple(float(v) for v in vals)
    if "radians" in block:
        rad_map = {str(k): float(v) for k, v in block["radians"].items()}
        return inspire_radians_by_name(rad_map)
    raise ValueError("hand block needs 'radians' or 'dds_normalized'")


def _load_yaml_poses(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pip install pyyaml (needed for --yaml)") from exc
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def _emit_keyframe(name: str, duration: float, left_arm: tuple[float, ...],
                   right_arm: tuple[float, ...], hand_q: tuple[float, ...], side: str) -> str:
    la = _round_tuple(left_arm)
    ra = _round_tuple(right_arm)
    hand = hand_call_from_dds(hand_q, side=side)
    return f'            GraspKeyframe("{name}", {duration}, {la}, {ra}, {hand}),'


def emit_grasp_sequence(data: dict[str, Any]) -> str:
    grasp_id = int(data["grasp_id"])
    label = str(data["label"])
    side = str(data.get("active_side", "left"))
    keyframes = data.get("keyframes", {})
    if not keyframes:
        raise ValueError("no keyframes in YAML")

    lines = [
        f"def _sequence_{grasp_id}() -> GraspSequence:",
        "    return GraspSequence(",
        f"        grasp_id={grasp_id},",
        f'        label="{label}",',
        f'        active_side="{side}",',
        "        keyframes=(",
    ]

    for name, kf in keyframes.items():
        duration = float(kf.get("duration_s", 3.0))
        if side == "left":
            left_arm = _arm_from_named(kf.get("left_arm", {}), LEFT_ARM_JOINT_NAMES)
            right_arm = _arm_from_named(kf.get("right_arm", {}), RIGHT_ARM_JOINT_NAMES) if "right_arm" in kf else left_arm
        else:
            right_arm = _arm_from_named(kf.get("right_arm", {}), RIGHT_ARM_JOINT_NAMES)
            left_arm = _arm_from_named(kf.get("left_arm", {}), LEFT_ARM_JOINT_NAMES) if "left_arm" in kf else right_arm

        hand_key = "left_hand" if side == "left" else "right_hand"
        hand_q = _hand_from_yaml_block(kf.get(hand_key), side)
        lines.append(_emit_keyframe(name, duration, left_arm, right_arm, hand_q, side))

    lines.extend([
        "        ),",
        "    )",
        "",
        f"# GRASPS[{grasp_id}] = _sequence_{grasp_id}(),",
    ])
    return "\n".join(lines)


def print_joint_list() -> None:
    print("# Left arm (rad) — copy Property > Joint Position into YAML left_arm:")
    for n in LEFT_ARM_JOINT_NAMES:
        print(f"  {n}")
    print("\n# Left Inspire hand (rad) — only proximal / yaw joints into YAML left_hand.radians:")
    for n in DDS_INSPIRE_JOINT_NAMES:
        if n.startswith("L_"):
            print(f"  {n}")
    print("\n# Order in grasp_library left_arm tuple:")
    print("  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw")
    print("\n# DDS hand order for _hand_left: pinky, ring, middle, index, thumb_p, thumb_y (0-1)")


def live_snapshot(timeout: float, interface: str | None, channel: int) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from .dds_env import configure_local_dds, dds_interface

    configure_local_dds()
    iface = dds_interface(interface)
    ChannelFactoryInitialize(channel, iface)

    low: dict[str, Any] = {}
    hand: dict[str, Any] = {}

    def on_low(msg: LowState_) -> None:
        low["q"] = [float(msg.motor_state[i].q) for i in range(29)]

    def on_hand(msg: MotorStates_) -> None:
        hand["q"] = [float(s.q) for s in msg.states[:12]]

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(on_low, 10)
    sub_hand = ChannelSubscriber("rt/inspire/state", MotorStates_)
    sub_hand.Init(on_hand, 10)

    deadline = time.time() + timeout
    while time.time() < deadline and ("q" not in low or "q" not in hand):
        time.sleep(0.05)

    if "q" not in low:
        raise TimeoutError("no rt/lowstate — is sim running with DDS?")
    if "q" not in hand:
        raise TimeoutError("no rt/inspire/state — use --enable_inspire_dds")

    q = low["q"]
    left_arm = tuple(q[i] for i in LEFT_ARM_JOINTS)
    right_arm = tuple(q[i] for i in RIGHT_ARM_JOINTS)
    hand_q = tuple(hand["q"][:12])
    return left_arm, right_arm, hand_q


def _save_session(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    session: list[dict[str, Any]] = []
    if path.exists():
        session = json.loads(path.read_text())
    session = [e for e in session if e.get("name") != entry["name"]]
    session.append(entry)
    path.write_text(json.dumps(session, indent=2))


def _emit_session(path: Path, grasp_id: int, label: str, side: str, order: list[str]) -> str:
    session = json.loads(path.read_text())
    by_name = {e["name"]: e for e in session}
    data = {
        "grasp_id": grasp_id,
        "label": label,
        "active_side": side,
        "keyframes": {},
    }
    for name in order:
        if name not in by_name:
            raise ValueError(f"phase '{name}' not captured; have {list(by_name)}")
        e = by_name[name]
        arm_key = "left_arm" if side == "left" else "right_arm"
        hand_key = "left_hand" if side == "left" else "right_hand"
        data["keyframes"][name] = {
            "duration_s": e["duration_s"],
            arm_key: dict(zip(
                LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES,
                e["left_arm" if side == "left" else "right_arm"],
            )),
            hand_key: {"dds_normalized": list(e["hand_q"])},
        }
    return emit_grasp_sequence(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint-list", action="store_true", help="Print joint names to find in Isaac GUI")
    p.add_argument("--yaml", type=Path, help="YAML pose file -> grasp_library snippet")
    p.add_argument("--live", action="store_true", help="Snapshot current pose from DDS topics")
    p.add_argument("--name", type=str, default="pose", help="Keyframe name (live / session)")
    p.add_argument("--duration", type=float, default=5.0, help="duration_s for this keyframe")
    p.add_argument("--side", type=str, default="left", choices=["left", "right"])
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--interface", type=str, default="auto")
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--session", type=Path, default=_SESSION_DEFAULT, help="JSON file for multi-phase capture")
    p.add_argument("--emit-session", action="store_true", help="Emit full grasp from saved session phases")
    p.add_argument("--grasp-id", type=int, default=1)
    p.add_argument("--label", type=str, default="custom")
    p.add_argument("--phases", type=str, default="approach,close,lift", help="Comma-separated phase order")
    args = p.parse_args(argv)

    if args.joint_list:
        print_joint_list()
        return 0

    if args.yaml:
        data = _load_yaml_poses(args.yaml)
        print(emit_grasp_sequence(data))
        return 0

    if args.emit_session:
        order = [x.strip() for x in args.phases.split(",") if x.strip()]
        if not args.session.exists():
            print(f"session file not found: {args.session}", file=sys.stderr)
            return 1
        print(_emit_session(args.session, args.grasp_id, args.label, args.side, order))
        return 0

    if args.live:
        iface = None if args.interface == "auto" else args.interface
        left_arm, right_arm, hand_q = live_snapshot(args.timeout, iface, args.channel)
        side = args.side
        la = left_arm if side == "left" else right_arm
        ra = right_arm if side == "left" else left_arm
        print(f"# live snapshot side={side}")
        print(_emit_keyframe(args.name, args.duration, left_arm, right_arm, hand_q, side))
        print(f"# left_arm  = {_round_tuple(left_arm)}")
        print(f"# right_arm = {_round_tuple(right_arm)}")
        print(f"# hand_q DDS norm = {tuple(round(v, 3) for v in hand_q)}")

        _save_session(args.session, {
            "name": args.name,
            "duration_s": args.duration,
            "side": side,
            "left_arm": list(left_arm),
            "right_arm": list(right_arm),
            "hand_q": list(hand_q),
        })
        print(f"# saved to {args.session}")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
