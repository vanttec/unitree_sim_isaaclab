"""CycloneDDS environment setup for local sim <-> bridge on one machine."""

from __future__ import annotations

import os

from dds.net_iface import detect_multicast_interface, resolve_dds_interface, validate_interface


def dds_interface(explicit: str | None = None) -> str | None:
    """Return network interface for ChannelFactoryInitialize, or None = autodetermine."""
    return resolve_dds_interface(explicit)


def configure_local_dds(force: bool = False) -> None:
    """Ensure UNITREE_DDS_INTERFACE is set for same-host sim + bridge."""
    env_iface = os.environ.get("UNITREE_DDS_INTERFACE", "")
    if env_iface and validate_interface(env_iface) and not force:
        return
    if env_iface and not validate_interface(env_iface):
        print(f"[dds] Stale UNITREE_DDS_INTERFACE={env_iface}, re-detecting")
    detected = detect_multicast_interface()
    if detected:
        os.environ["UNITREE_DDS_INTERFACE"] = detected
    else:
        os.environ.pop("UNITREE_DDS_INTERFACE", None)
    os.environ.pop("CYCLONEDDS_URI", None)
