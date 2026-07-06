#!/usr/bin/env python3
"""Record, replay, and list teleop trajectories (slots 1-5). See doc/RUNBOOK.md."""

from __future__ import annotations

import argparse
import sys
import threading

from .dds_client import UnitreeDdsClient
from .episode_to_trajectory import convert_episode_json
from .trajectory_executor import TrajectoryExecutor, DEFAULT_HAND_GRIP_BOOST
from .trajectory_inspect import inspect_slot
from .trajectory_io import list_trajectories, trim_trajectory
from .trajectory_recorder import TrajectoryRecorder


def _cmd_list() -> int:
    for row in list_trajectories():
        if row["frames"] == 0:
            print(f"  {row['slot']}: (empty)")
        else:
            print(
                f"  {row['slot']}: {row['label']} — "
                f"{row['frames']} frames, {row['duration_s']:.1f}s @ {row['hz']:.0f} Hz"
            )
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    rec = TrajectoryRecorder(
        domain_channel=args.channel,
        network_interface=None if args.interface == "auto" else args.interface,
        local_dds=not args.no_local_dds,
    )
    print(f"[traj] Waiting for DDS (slot {args.slot})...")
    rec.wait_for_state(timeout_s=args.timeout)
    print("[traj] DDS ready. Open teleop [r], hands open → ENTER to record → move → ENTER to stop.")
    print("[traj] Record at 100 Hz for best finger replay (avoid --hz 30).")
    input()
    stop_flag = threading.Event()

    def _wait_stop():
        input()
        stop_flag.set()

    threading.Thread(target=_wait_stop, daemon=True).start()
    print(f"[traj] Recording slot {args.slot} @ {args.hz} Hz...")
    rec.record_until(stop=stop_flag.is_set, hz=args.hz)
    path = rec.save_slot(args.slot, label=args.label or f"trajectory_{args.slot}")
    print(f"[traj] Saved {path}")
    inspect_slot(args.slot)
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    from .trajectory_registry import get_slot

    iface = None if args.interface == "auto" else args.interface
    client = UnitreeDdsClient(
        domain_channel=args.channel,
        network_interface=iface,
        local_dds=not args.no_local_dds,
    )
    client.wait_for_state(timeout_s=args.timeout)
    freeze = get_slot(args.slot).freeze_left_arm or args.freeze_left_arm
    TrajectoryExecutor(
        client,
        hand_kp=args.hand_kp,
        hand_kd=args.hand_kd,
        hand_grip_boost=args.hand_grip_boost,
    ).run(
        args.slot,
        loop=args.loop,
        ramp_s=args.ramp_s,
        reset_object=args.reset_object,
        reset_channel=args.channel,
        freeze_left_arm=freeze,
    )
    return 0


def _cmd_trim(args: argparse.Namespace) -> int:
    path = trim_trajectory(args.slot, tail_s=args.tail_s, head_s=args.head_s)
    print(f"[traj] Trimmed slot {args.slot} -> {path}")
    inspect_slot(args.slot)
    return 0


def _cmd_from_episode(args: argparse.Namespace) -> int:
    path = convert_episode_json(args.episode_json, args.slot, hz=args.hz, label=args.label)
    print(f"[traj] Converted -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Teleop trajectory slots 1-4")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List trajectory slots")

    p_ins = sub.add_parser("inspect", help="Diagnose a recorded trajectory")
    p_ins.add_argument("slot", type=int, choices=[1, 2, 3, 4, 5])

    p_rec = sub.add_parser("record", help="Record from live DDS while teleoperating")
    p_rec.add_argument("--slot", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p_rec.add_argument("--hz", type=float, default=100.0, help="Match teleop Inspire rate (100 Hz)")
    p_rec.add_argument("--label", type=str, default="")
    p_rec.add_argument("--channel", type=int, default=1)
    p_rec.add_argument("--interface", type=str, default="auto")
    p_rec.add_argument("--no-local-dds", action="store_true")
    p_rec.add_argument("--timeout", type=float, default=30.0)

    p_play = sub.add_parser("play", help="Replay trajectory over DDS")
    p_play.add_argument("slot", type=int, choices=[1, 2, 3, 4, 5])
    p_play.add_argument("--hand-kp", type=float, default=0.0, help="Inspire finger kp during replay (teleop uses 0)")
    p_play.add_argument("--hand-kd", type=float, default=0.0)
    p_play.add_argument(
        "--hand-grip-boost",
        type=float,
        default=DEFAULT_HAND_GRIP_BOOST,
        help="Extra finger closure 0-1 (replay only)",
    )
    p_play.add_argument("--loop", action="store_true")
    p_play.add_argument("--reset-object", action="store_true", help="Reset object before replay")
    p_play.add_argument("--ramp-s", type=float, default=3.0, help="Seconds to interpolate sim pose → recording frame 0")
    p_play.add_argument("--channel", type=int, default=1)
    p_play.add_argument("--interface", type=str, default="auto")
    p_play.add_argument("--no-local-dds", action="store_true")
    p_play.add_argument("--freeze-left-arm", action="store_true", help="Hold left arm at frame-0 pose during replay")
    p_play.add_argument("--timeout", type=float, default=30.0)

    p_trim = sub.add_parser("trim", help="Remove seconds from start/end of a trajectory")
    p_trim.add_argument("slot", type=int, choices=[1, 2, 3, 4, 5])
    p_trim.add_argument("--tail-s", type=float, default=0.0, help="Seconds to cut from the end")
    p_trim.add_argument("--head-s", type=float, default=0.0, help="Seconds to cut from the start")

    p_ep = sub.add_parser("from-episode", help="Build .npz from xr_teleoperate episode data.json")
    p_ep.add_argument("episode_json", type=str)
    p_ep.add_argument("--slot", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p_ep.add_argument("--hz", type=float, default=30.0)
    p_ep.add_argument("--label", type=str, default="")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.command == "inspect":
        inspect_slot(args.slot)
        return 0
    if args.command == "record":
        return _cmd_record(args)
    if args.command == "play":
        return _cmd_play(args)
    if args.command == "trim":
        return _cmd_trim(args)
    if args.command == "from-episode":
        from pathlib import Path
        args.episode_json = Path(args.episode_json)
        return _cmd_from_episode(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
