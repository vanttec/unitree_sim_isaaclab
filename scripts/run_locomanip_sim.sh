#!/usr/bin/env bash
# Loco-manipulation teleop in sim: GR00T-WBC legs (keyboard) + Quest arms/hands + episode recorder.
# Usage:
#   ./scripts/run_locomanip_sim.sh                 # GUI
#   HEADLESS=1 ./scripts/run_locomanip_sim.sh      # no Isaac window (Quest still gets the cameras)
#   OUT=data/locomanip_pick ./scripts/run_locomanip_sim.sh
#   DEVICE=cuda RENDER_INTERVAL=2 ./scripts/run_locomanip_sim.sh
#
# Physics defaults to CPU: for ONE robot PhysX-CPU is ~4x faster than the GPU pipeline (13 vs 53 ms per
# control step measured with WH_PROFILE=1 on the RTX 5070 Ti laptop); the GPU still renders the cameras.
# RENDER_INTERVAL=4 renders/refreshes cameras every 4th control step: ~32 Hz loop (~0.65x real time)
# instead of ~8 Hz rendering every step.
# Then, in other terminals: xr_teleoperate (--sim), send_commands_keyboard.py, python -m locomanip.cli keys.
# Full steps: doc/LOCOMANIP.md
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-data/locomanip}"
DEVICE="${DEVICE:-cpu}"
RENDER_INTERVAL="${RENDER_INTERVAL:-4}"
CAMERAS="${CAMERAS:-front_camera,left_wrist_camera,right_wrist_camera}"
HEADLESS_FLAG=""
[[ "${HEADLESS:-0}" == "1" ]] && HEADLESS_FLAG="--headless"

source grasp_bridge/setup_local_dds.sh
echo "Episodes -> $OUT"
PYTHONUNBUFFERED=1 python sim_main.py --device "$DEVICE" --enable_cameras $HEADLESS_FLAG \
  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody \
  --enable_inspire_dds --robot_type g129 \
  --camera_include "$CAMERAS" --render_interval "$RENDER_INTERVAL" \
  --wbc_internal \
  --locomanip_record --locomanip_dir "$OUT"
