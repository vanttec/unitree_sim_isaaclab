"""CycloneDDS environment setup for local sim <-> bridge on one machine."""

from __future__ import annotations

import os
import subprocess


def dds_interface(explicit: str | None = None) -> str | None:
    """Return network interface for ChannelFactoryInitialize, or None = autodetermine."""
    if explicit is not None and explicit not in ("", "auto"):
        return explicit
    env_iface = os.environ.get("UNITREE_DDS_INTERFACE")
    if env_iface and env_iface not in ("", "auto"):
        return env_iface
    return None


def detect_multicast_interface() -> str | None:
    try:
        out = subprocess.check_output(["ip", "-br", "link", "show"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    skip_prefixes = ("lo", "docker", "zt", "veth", "br-")
    for line in out.splitlines():
        name = line.split()[0] if line.split() else ""
        if not name or "UP" not in line or "MULTICAST" not in line:
            continue
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        return name
    return None


def configure_local_dds(force: bool = False) -> None:
    """Ensure UNITREE_DDS_INTERFACE is set for same-host sim + bridge."""
    if os.environ.get("UNITREE_DDS_INTERFACE") and not force:
        return
    detected = detect_multicast_interface()
    if detected:
        os.environ["UNITREE_DDS_INTERFACE"] = detected
    os.environ.pop("CYCLONEDDS_URI", None)
