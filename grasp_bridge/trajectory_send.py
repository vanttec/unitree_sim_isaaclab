#!/usr/bin/env python3
"""Send TCP trajectory commands to trajectory_socket_server, or replay locally.

TCP commands (uint32 LE over socket):
    1 = container, 2 = stick, 4 = tennisball, 6 = cardsdeck

Keyboard dummy:
    python -m grasp_bridge.trajectory_send --interactive

One-shot TCP:
    python -m grasp_bridge.trajectory_send 6

Local replay (maps TCP cmd -> internal slot):
    python -m grasp_bridge.trajectory_send 6 --local
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys

from .dds_client import UnitreeDdsClient
from .trajectory_executor import TrajectoryExecutor, play_trajectory_slot, DEFAULT_HAND_GRIP_BOOST
from .trajectory_registry import (
    TCP_TRAJECTORY_CMD,
    get_slot,
    slot_from_tcp_cmd,
    slot_has_recording,
    tcp_cmd_help,
)


def send_tcp_cmd(host: str, port: int, cmd: int, timeout: float = 120.0) -> str:
    slot_from_tcp_cmd(cmd)  # validate before connect
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(struct.pack("<I", cmd))
        reply = sock.recv(4)
    return reply.decode("ascii", errors="replace")


def run_local(
    slot: int,
    *,
    ramp_s: float,
    reset_object: bool,
    hand_kp: float,
    hand_kd: float,
    hand_grip_boost: float,
    channel: int,
    interface: str | None,
    local_dds: bool,
    timeout: float,
) -> None:
    client = UnitreeDdsClient(
        domain_channel=channel,
        network_interface=interface,
        local_dds=local_dds,
    )
    client.wait_for_state(timeout_s=timeout)
    executor = TrajectoryExecutor(
        client, hand_kp=hand_kp, hand_kd=hand_kd, hand_grip_boost=hand_grip_boost
    )
    play_trajectory_slot(
        executor,
        slot,
        ramp_s=ramp_s,
        reset_object=reset_object,
        reset_channel=channel,
    )


def interactive_loop(args: argparse.Namespace) -> int:
    valid = sorted(TCP_TRAJECTORY_CMD)
    print("Trajectory dummy — TCP cmds:", ", ".join(str(c) for c in valid), "| q quit")
    print(tcp_cmd_help())
    for tcp, slot in sorted(TCP_TRAJECTORY_CMD.items()):
        info = get_slot(slot)
        rec = "OK" if slot_has_recording(slot) else "empty"
        print(f"  tcp {tcp}: {info.label} -> traj slot {slot} ({rec}) env: {info.sim_env}")
    print(f"Target: {'local DDS' if args.local else f'{args.host}:{args.port} TCP'}")
    print()

    while True:
        try:
            line = input("tcp> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if line in ("q", "quit", "exit"):
            return 0
        try:
            cmd = int(line)
        except ValueError:
            print(f"Use {valid} or q")
            continue
        if cmd not in TCP_TRAJECTORY_CMD:
            print(f"Unknown TCP cmd {cmd}; valid: {valid}")
            continue

        slot = slot_from_tcp_cmd(cmd)
        if not slot_has_recording(slot):
            print(f"Slot {slot} has no traj_{slot}.npz — record first.")
            continue

        info = get_slot(slot)
        print(f"→ tcp {cmd} / slot {slot} / {info.label} (sim: ./scripts/run_inspire_teleop_env.sh {info.sim_env})")

        try:
            if args.local:
                run_local(
                    slot,
                    ramp_s=args.ramp_s,
                    reset_object=args.reset_object,
                    hand_kp=args.hand_kp,
                    hand_kd=args.hand_kd,
                    hand_grip_boost=args.hand_grip_boost,
                    channel=args.channel,
                    interface=None if args.interface == "auto" else args.interface,
                    local_dds=not args.no_local_dds,
                    timeout=args.timeout,
                )
                print("OK (local)")
            else:
                reply = send_tcp_cmd(args.host, args.port, cmd, timeout=args.socket_timeout)
                print(reply)
        except Exception as exc:
            print(f"ERROR: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send TCP trajectory cmd (1/2/4/6) or replay locally")
    parser.add_argument("cmd", nargs="?", type=int, help="TCP cmd: 1=container, 2=stick, 4=tennisball, 6=cardsdeck")
    parser.add_argument("--interactive", "-i", action="store_true", help="Keyboard loop")
    parser.add_argument("--local", action="store_true", help="Replay via DDS (maps TCP cmd to slot)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    parser.add_argument("--ramp-s", type=float, default=2.0)
    parser.add_argument("--reset-object", action="store_true", help="Reset object before replay")
    parser.add_argument("--hand-kp", type=float, default=0.0)
    parser.add_argument("--hand-kd", type=float, default=0.0)
    parser.add_argument("--hand-grip-boost", type=float, default=DEFAULT_HAND_GRIP_BOOST)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--interface", type=str, default="auto")
    parser.add_argument("--no-local-dds", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    if args.interactive:
        return interactive_loop(args)

    if args.cmd is None:
        parser.error("TCP cmd required (1/2/4/6), or use --interactive")

    try:
        slot = slot_from_tcp_cmd(args.cmd)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not slot_has_recording(slot):
        print(f"ERROR: no recording in slot {slot} (traj_{slot}.npz)", file=sys.stderr)
        return 1

    try:
        if args.local:
            run_local(
                slot,
                ramp_s=args.ramp_s,
                reset_object=args.reset_object,
                hand_kp=args.hand_kp,
                hand_kd=args.hand_kd,
                hand_grip_boost=args.hand_grip_boost,
                channel=args.channel,
                interface=None if args.interface == "auto" else args.interface,
                local_dds=not args.no_local_dds,
                timeout=args.timeout,
            )
            print("OK")
        else:
            print(send_tcp_cmd(args.host, args.port, args.cmd, timeout=args.socket_timeout))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
