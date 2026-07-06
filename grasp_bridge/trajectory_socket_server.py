#!/usr/bin/env python3
"""TCP server: uint32 trajectory cmd → reset + replay slot. Port 5556."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading

from .dds_client import UnitreeDdsClient
from .trajectory_executor import TrajectoryExecutor, play_trajectory_slot, DEFAULT_HAND_GRIP_BOOST
from .trajectory_registry import (
    get_slot,
    slot_from_tcp_cmd,
    slot_has_recording,
    tcp_cmd_help,
)


def handle_client(
    conn: socket.socket,
    executor: TrajectoryExecutor,
    busy: threading.Lock,
    *,
    ramp_s: float,
    reset_object: bool,
    reset_channel: int,
) -> None:
    try:
        data = conn.recv(4)
        if len(data) != 4:
            conn.sendall(b"ERROR")
            return

        cmd = struct.unpack("<I", data)[0]
        peer = conn.getpeername()
        try:
            slot = slot_from_tcp_cmd(cmd)
        except ValueError as exc:
            print(f"[traj-socket] {exc} (from {peer})")
            conn.sendall(b"ERROR")
            return

        info = get_slot(slot)
        print(f"[traj-socket] TCP cmd={cmd} -> slot {slot} ({info.label}) from {peer}")
        if not slot_has_recording(slot):
            print(f"[traj-socket] No recording in slot {slot} (traj_{slot}.npz)")
            conn.sendall(b"ERROR")
            return

        if not busy.acquire(blocking=False):
            print("[traj-socket] Replay already running, rejecting command")
            conn.sendall(b"ERROR")
            return

        try:
            play_trajectory_slot(
                executor,
                slot,
                ramp_s=ramp_s,
                reset_object=reset_object,
                reset_channel=reset_channel,
            )
            conn.sendall(b"OK")
        finally:
            busy.release()
    except Exception as exc:
        print(f"[traj-socket] Error: {exc}")
        try:
            conn.sendall(b"ERROR")
        except OSError:
            pass
    finally:
        conn.close()


def run_server(
    host: str = "0.0.0.0",
    port: int = 5556,
    *,
    ramp_s: float = 2.0,
    reset_object: bool = False,
    hand_kp: float = 0.0,
    hand_kd: float = 0.0,
    hand_grip_boost: float = DEFAULT_HAND_GRIP_BOOST,
    channel: int = 1,
    interface: str | None = None,
    local_dds: bool = True,
    timeout: float = 30.0,
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
    busy = threading.Lock()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)

    print(f"[traj-socket] Listening on {host}:{port}")
    print(f"[traj-socket] TCP mapping: {tcp_cmd_help()}")
    print("[traj-socket] Stop teleop before sending a replay command.")

    try:
        while True:
            conn, addr = sock.accept()
            print(f"[traj-socket] Connected: {addr}")
            threading.Thread(
                target=handle_client,
                args=(conn, executor, busy),
                kwargs={
                    "ramp_s": ramp_s,
                    "reset_object": reset_object,
                    "reset_channel": channel,
                },
                daemon=True,
            ).start()
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Teleop trajectory replay TCP server (slots 1-5)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
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

    iface = None if args.interface == "auto" else args.interface
    run_server(
        host=args.host,
        port=args.port,
        ramp_s=args.ramp_s,
        reset_object=args.reset_object,
        hand_kp=args.hand_kp,
        hand_kd=args.hand_kd,
        hand_grip_boost=args.hand_grip_boost,
        channel=args.channel,
        interface=iface,
        local_dds=not args.no_local_dds,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
