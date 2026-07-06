#!/usr/bin/env python3
"""TCP server: uint32 grasp_id (1-6) → replay semantic grasp. Port 5555."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading

from .dds_client import UnitreeDdsClient
from .executor import GraspExecutor
from .grasp_library import GRASPS


def handle_client(conn: socket.socket, executor: GraspExecutor, busy: threading.Lock) -> None:
    try:
        data = conn.recv(4)
        if len(data) != 4:
            conn.sendall(b"ERROR")
            return

        grasp_id = struct.unpack("<I", data)[0]
        print(f"[socket] Received grasp_id={grasp_id} from {conn.getpeername()}")

        if grasp_id not in GRASPS:
            print(f"[socket] Invalid grasp_id {grasp_id}; use 1-6")
            conn.sendall(b"ERROR")
            return

        if not busy.acquire(blocking=False):
            print("[socket] Grasp already running, rejecting command")
            conn.sendall(b"ERROR")
            return

        try:
            executor.run(grasp_id)
            conn.sendall(b"OK")
        finally:
            busy.release()
    except Exception as exc:
        print(f"[socket] Error: {exc}")
        try:
            conn.sendall(b"ERROR")
        except OSError:
            pass
    finally:
        conn.close()


def run_server(
    host: str = "0.0.0.0",
    port: int = 5555,
    control_hz: float = 50.0,
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
    executor = GraspExecutor(client, control_hz=control_hz)
    busy = threading.Lock()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)

    labels = ", ".join(f"{k}={v.label}" for k, v in sorted(GRASPS.items()))
    print(f"[socket] Listening on {host}:{port}")
    print(f"[socket] Send 4-byte uint32 LE (1-6). Grasps: {labels}")

    try:
        while True:
            conn, addr = sock.accept()
            print(f"[socket] Connected: {addr}")
            threading.Thread(
                target=handle_client,
                args=(conn, executor, busy),
                daemon=True,
            ).start()
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grasp bridge TCP server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5555, help="TCP port to listen on")
    parser.add_argument("--hz", type=float, default=50.0, help="Control rate")
    parser.add_argument("--channel", type=int, default=1, help="DDS domain id")
    parser.add_argument("--interface", type=str, default="auto", help="Network iface (auto, or e.g. wlp131s0f0)")
    parser.add_argument("--no-local-dds", action="store_true", help="Do not force loopback CYCLONEDDS_URI (real robot LAN)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for rt/lowstate")
    args = parser.parse_args(argv)

    iface = None if args.interface == "auto" else args.interface
    run_server(
        host=args.host,
        port=args.port,
        control_hz=args.hz,
        channel=args.channel,
        interface=iface,
        local_dds=not args.no_local_dds,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
