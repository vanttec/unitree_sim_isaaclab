#!/usr/bin/env bash
# Trajectory replay TCP server (slots 1–5). Source DDS, then listen on :5556.
#
# Usage:
#   ./scripts/start_traj_socket.sh
#   ./scripts/start_traj_socket.sh --port 5556
#
# In another terminal (after sim is up, teleop stopped):
#   python -m grasp_bridge.trajectory_send --interactive

set -euo pipefail
cd "$(dirname "$0")/.."
source grasp_bridge/setup_local_dds.sh
exec python -m grasp_bridge.trajectory_socket_server "$@"
