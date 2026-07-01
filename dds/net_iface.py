"""Pick a working network interface for CycloneDDS multicast."""

from __future__ import annotations

import subprocess


def _link_rows() -> list[str]:
    try:
        out = subprocess.check_output(["ip", "-br", "link", "show"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def validate_interface(name: str) -> bool:
    """True if iface exists, kernel state is UP, and link has carrier (LOWER_UP)."""
    if not name:
        return False
    skip_prefixes = ("lo", "docker", "zt", "veth", "br-")
    if any(name.startswith(p) for p in skip_prefixes):
        return False
    for line in _link_rows():
        parts = line.split()
        if not parts or parts[0] != name:
            continue
        if len(parts) < 2 or parts[1] != "UP":
            return False
        return "LOWER_UP" in line
    return False


def detect_multicast_interface() -> str | None:
    """First UP interface with carrier and MULTICAST (WiFi/Ethernet)."""
    skip_prefixes = ("lo", "docker", "zt", "veth", "br-")
    for line in _link_rows():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if state != "UP" or "MULTICAST" not in line or "LOWER_UP" not in line:
            continue
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        return name
    return None


def resolve_dds_interface(explicit: str | None = None) -> str | None:
    """Return iface for ChannelFactoryInitialize, or None = autodetermine."""
    import os

    candidates: list[str] = []
    if explicit and explicit not in ("", "auto"):
        candidates.append(explicit)
    env_iface = os.environ.get("UNITREE_DDS_INTERFACE", "")
    if env_iface and env_iface not in ("", "auto"):
        candidates.append(env_iface)

    for name in candidates:
        if validate_interface(name):
            return name
        print(f"[dds] Interface '{name}' unavailable (down or no carrier), skipping")

    detected = detect_multicast_interface()
    if detected:
        print(f"[dds] Auto-detected interface: {detected}")
        return detected
    return None
