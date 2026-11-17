"""Start / stop / discard loco-manipulation episodes in a running sim (sim_main.py --locomanip_record).

    python -m locomanip.cli start pick_cylinder_walk     # begin an episode
    python -m locomanip.cli stop                         # save it
    python -m locomanip.cli discard                      # throw it away
    python -m locomanip.cli keys                         # interactive: r=start  f=stop/save  g=discard  Ctrl-C=quit

Keys avoid w/a/s/d/z/x/c/q: send_commands_keyboard.py listens to the whole X keyboard (pynput).

Same DDS domain (1) and iface as the sim; uses grasp_bridge's iface helper, so source
grasp_bridge/setup_local_dds.sh first when teleoperating over a real NIC.
"""
from __future__ import annotations

import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from .g1_joints import RECORD_TOPIC


def _publisher(iface):
    try:
        from grasp_bridge.dds_env import dds_interface
        iface = dds_interface(iface)
    except Exception:
        pass
    ChannelFactoryInitialize(1, iface) if iface else ChannelFactoryInitialize(1)
    pub = ChannelPublisher(RECORD_TOPIC, String_)
    pub.Init()
    time.sleep(0.5)   # let discovery match the sim's subscriber before the first write
    return pub


def send(pub, text):
    msg = String_(data=text)
    for _ in range(3):   # best-effort topic: repeat; the recorder ignores a repeated stop/discard
        pub.Write(msg)
        time.sleep(0.05)
    print(f"-> {text}")


def keys(pub, label):
    import termios
    import tty
    n = 0
    print("r = start   f = stop (save)   g = discard   Ctrl-C = quit")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            c = sys.stdin.read(1).lower()
            if c == "r":
                n += 1
                send(pub, f"start {label}_{n:03d}")
            elif c == "f":
                send(pub, "stop")
            elif c == "g":
                send(pub, "discard")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop", "discard", "keys"])
    p.add_argument("label", nargs="?", default="episode")
    p.add_argument("--iface", default=None, help="DDS network interface (default: same rule as grasp_bridge)")
    a = p.parse_args()
    pub = _publisher(a.iface)
    if a.action == "keys":
        keys(pub, a.label)
    elif a.action == "start":
        send(pub, f"start {a.label}")
    else:
        send(pub, a.action)


if __name__ == "__main__":
    main()
