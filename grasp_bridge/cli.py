#!/usr/bin/env python3
"""CLI: run semantic grasp 1-6 over DDS."""

from __future__ import annotations

import argparse
import sys

from .dds_client import UnitreeDdsClient
from .executor import GraspExecutor
from .grasp_library import GRASPS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unitree G1 Inspire grasp bridge (DDS CLI)")
    parser.add_argument("grasp_id", nargs="?", type=int, help="Grasp type 1-6")
    parser.add_argument("--list", action="store_true", help="List available grasps")
    parser.add_argument("--hz", type=float, default=50.0, help="Control rate (default: 50)")
    parser.add_argument("--channel", type=int, default=1, help="DDS domain id (sim uses 1)")
    parser.add_argument("--interface", type=str, default="auto", help="Network iface (auto, or e.g. wlp131s0f0)")
    parser.add_argument("--no-local-dds", action="store_true", help="Do not force loopback CYCLONEDDS_URI (real robot LAN)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for rt/lowstate")
    args = parser.parse_args(argv)

    if args.list:
        for gid, seq in sorted(GRASPS.items()):
            print(f"  {gid}: {seq.label}")
        return 0

    if args.grasp_id is None:
        parser.error("grasp_id required (1-6), or use --list")

    iface = None if args.interface == "auto" else args.interface
    client = UnitreeDdsClient(
        domain_channel=args.channel,
        network_interface=iface,
        local_dds=not args.no_local_dds,
    )
    client.wait_for_state(timeout_s=args.timeout)
    executor = GraspExecutor(client, control_hz=args.hz)
    executor.run(args.grasp_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
