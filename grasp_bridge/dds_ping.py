#!/usr/bin/env python3
"""Listen for rt/lowstate — quick DDS connectivity check."""

from __future__ import annotations

import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from .dds_env import configure_local_dds, dds_interface


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping Unitree DDS rt/lowstate")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--interface", type=str, default="auto")
    parser.add_argument("--channel", type=int, default=1)
    args = parser.parse_args()

    configure_local_dds()
    iface = None if args.interface == "auto" else args.interface
    iface = dds_interface(iface)
    print(f"[dds_ping] iface={iface or 'autodetermine'} domain={args.channel}")

    count = 0

    def on_state(msg: LowState_) -> None:
        nonlocal count
        count += 1
        if count == 1:
            q0 = msg.motor_state[0].q
            print(f"[dds_ping] FIRST rt/lowstate: motor[0].q={q0:.4f}")

    ChannelFactoryInitialize(args.channel, iface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    deadline = time.time() + args.timeout
    while time.time() < deadline and count == 0:
        time.sleep(0.1)

    if count == 0:
        print("[dds_ping] FAIL — no rt/lowstate (restart sim with setup_local_dds.sh sourced first)")
        return 1

    time.sleep(1.0)
    print(f"[dds_ping] OK — {count}+ messages received")
    return 0


if __name__ == "__main__":
    sys.exit(main())
