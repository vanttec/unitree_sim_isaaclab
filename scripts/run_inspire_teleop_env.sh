#!/usr/bin/env bash
# Launch Isaac Sim + Inspire for xr_teleoperate + trajectory slots 1–5.
# Usage:
#   ./scripts/run_inspire_teleop_env.sh coin        # slot 1
#   ./scripts/run_inspire_teleop_env.sh stick       # slot 2
#   ./scripts/run_inspire_teleop_env.sh tennisball  # slot 3
#   ./scripts/run_inspire_teleop_env.sh cardsdeck   # slot 4
#   ./scripts/run_inspire_teleop_env.sh container   # slot 5
#   ./scripts/run_inspire_teleop_env.sh redblock | cylinder
#
# Replay via socket (other terminal): ./scripts/start_traj_socket.sh
#
# Default behaviour renders every step with all 3 cameras (original setup).
# Optional lightweight mode only if you ever hit a perf wall (GPU should handle full):
#   FAST=1               only front camera + render every 2 steps
#   HEADLESS=1           no Isaac GUI window (cameras still render offscreen for the Quest)
#   RENDER_INTERVAL=N    render cameras every N control steps
#   CAM_WRITE_INTERVAL=N push camera frame to ZMQ every N steps
#   CAMERAS="a,b"        override which cameras are enabled
#
# Normal (default):   ./scripts/run_inspire_teleop_env.sh cardsdeck
# Lightweight:        FAST=1 ./scripts/run_inspire_teleop_env.sh cardsdeck
# Fastest (VR only):  FAST=1 HEADLESS=1 ./scripts/run_inspire_teleop_env.sh cardsdeck

set -euo pipefail

ENV_KEY="${1:-coin}"
DEVICE="${2:-cuda}"

# --- original defaults: render every step, all cameras ---
RENDER_INTERVAL="${RENDER_INTERVAL:-1}"
CAM_WRITE_INTERVAL="${CAM_WRITE_INTERVAL:-1}"
CAMERAS="${CAMERAS:-front_camera,left_wrist_camera,right_wrist_camera}"

# Opt-in lightweight mode.
if [[ "${FAST:-0}" == "1" ]]; then
  RENDER_INTERVAL="${RENDER_INTERVAL_FAST:-2}"
  CAM_WRITE_INTERVAL="${CAM_WRITE_INTERVAL_FAST:-2}"
  CAMERAS="front_camera"
fi

# Opt-in headless: skip the Isaac GUI viewport render (big win, cameras still stream).
HEADLESS_FLAG=""
if [[ "${HEADLESS:-0}" == "1" ]]; then
  HEADLESS_FLAG="--headless"
fi

case "$ENV_KEY" in
  coin)       TASK="Isaac-PickPlace-Coin-G129-Inspire-Joint" ;;
  stick)      TASK="Isaac-PickPlace-Stick-G129-Inspire-Joint" ;;
  tennisball) TASK="Isaac-PickPlace-TennisBall-G129-Inspire-Joint" ;;
  cardsdeck)  TASK="Isaac-PickPlace-CardsDeck-G129-Inspire-Joint" ;;
  container)  TASK="Isaac-PickPlace-Container-G129-Inspire-Joint" ;;
  redblock)   TASK="Isaac-PickPlace-RedBlock-G129-Inspire-Joint" ;;
  cylinder)   TASK="Isaac-PickPlace-Cylinder-G129-Inspire-Joint" ;;
  *)
    echo "Unknown env: $ENV_KEY"
    echo "Options: coin | stick | tennisball | cardsdeck | container | redblock | cylinder"
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."
echo "Task: $TASK"
echo "Device: $DEVICE"
echo "Render interval: $RENDER_INTERVAL | Camera write interval: $CAM_WRITE_INTERVAL"
echo "Cameras: $CAMERAS"
echo "Headless: ${HEADLESS:-0}"

python sim_main.py --device "$DEVICE" --enable_cameras $HEADLESS_FLAG \
  --task "$TASK" \
  --enable_inspire_dds --robot_type g129 \
  --camera_include "$CAMERAS" \
  --camera_write_interval "$CAM_WRITE_INTERVAL" \
  --render_interval "$RENDER_INTERVAL"
