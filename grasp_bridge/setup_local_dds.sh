#!/usr/bin/env bash
# Source in BOTH terminals before sim_main.py and grasp_bridge.cli:
#   source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh

IFACE="${UNITREE_DDS_INTERFACE:-$(ip -br link show | awk '
  $0 ~ /UP/ && $0 ~ /MULTICAST/ {
    name=$1
    if (name != "lo" && name !~ /^docker/ && name !~ /^zt/ && name !~ /^veth/ && name !~ /^br-/) {
      print name; exit
    }
  }
')}"

if [[ -z "$IFACE" ]]; then
  echo "ERROR: no multicast-capable network interface found." >&2
  echo "Set manually: export UNITREE_DDS_INTERFACE=wlp131s0f0" >&2
  return 1 2>/dev/null || exit 1
fi

export UNITREE_DDS_INTERFACE="$IFACE"
unset CYCLONEDDS_URI

echo "UNITREE_DDS_INTERFACE=${UNITREE_DDS_INTERFACE}  (source this in BOTH sim + bridge terminals)"
