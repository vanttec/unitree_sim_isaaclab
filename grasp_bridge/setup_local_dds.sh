#!/usr/bin/env bash
# Source in BOTH terminals before sim_main.py and grasp_bridge.cli:
#   source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh

# Require kernel state UP ($2) + LOWER_UP (real link), not just the IFF_UP flag on a dead USB NIC.
IFACE="${UNITREE_DDS_INTERFACE:-$(ip -br link show | awk '
  $2 == "UP" && $0 ~ /MULTICAST/ && $0 ~ /LOWER_UP/ {
    name=$1
    if (name != "lo" && name !~ /^docker/ && name !~ /^zt/ && name !~ /^veth/ && name !~ /^br-/) {
      print name; exit
    }
  }
')}"

# Stale env from a previous session (e.g. unplugged USB ethernet)?
if [[ -n "${UNITREE_DDS_INTERFACE:-}" ]] && ! ip -br link show "$UNITREE_DDS_INTERFACE" 2>/dev/null | awk '$2=="UP" && /LOWER_UP/ {found=1} END{exit !found}'; then
  echo "WARN: UNITREE_DDS_INTERFACE=$UNITREE_DDS_INTERFACE is down; re-detecting..." >&2
  unset UNITREE_DDS_INTERFACE
  IFACE="$(ip -br link show | awk '
    $2 == "UP" && $0 ~ /MULTICAST/ && $0 ~ /LOWER_UP/ {
      name=$1
      if (name != "lo" && name !~ /^docker/ && name !~ /^zt/ && name !~ /^veth/ && name !~ /^br-/) {
        print name; exit
      }
    }
  ')"
fi

if [[ -z "$IFACE" ]]; then
  echo "ERROR: no multicast-capable network interface found." >&2
  echo "Set manually: export UNITREE_DDS_INTERFACE=wlp131s0f0" >&2
  return 1 2>/dev/null || exit 1
fi

export UNITREE_DDS_INTERFACE="$IFACE"
unset CYCLONEDDS_URI

echo "UNITREE_DDS_INTERFACE=${UNITREE_DDS_INTERFACE}  (source this in BOTH sim + bridge terminals)"
