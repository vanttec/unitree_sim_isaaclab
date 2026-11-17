"""Recorded episodes -> SONIC motion_lib PKL.

    python -m locomanip.export data/locomanip --out data/locomanip_sonic

1. collects every <episode>/motion_50hz.csv into <out>/csv/<episode>.csv (Bones-SEED flat CSVs)
2. runs SONIC's own converter on them (gear_sonic/data_process/convert_soma_csv_to_motion_lib.py)
   -> <out>/locomanip_motion_lib.pkl at 50 fps, ready to mix into a SONIC fine-tune

Skip step 2 with --no-convert (e.g. to convert on the training machine instead).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

SONIC_ROOT = os.environ.get("SONIC_ROOT", os.path.expanduser("~/GR00T-WholeBodyControl"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes", help="directory holding <stamp>_<label>/ episode dirs")
    p.add_argument("--out", required=True)
    p.add_argument("--min-seconds", type=float, default=2.0, help="skip episodes shorter than this")
    p.add_argument("--sonic-python", default=sys.executable, help="needs numpy, scipy, joblib, pandas")
    p.add_argument("--no-convert", action="store_true")
    a = p.parse_args()

    csv_dir = os.path.join(a.out, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    kept, total_s = 0, 0.0
    for meta_f in sorted(glob.glob(os.path.join(a.episodes, "*", "meta.json"))):
        ep = os.path.dirname(meta_f)
        meta = json.load(open(meta_f))
        if meta["duration_sim_s"] < a.min_seconds:
            print(f"skip {os.path.basename(ep)} ({meta['duration_sim_s']:.1f} s)")
            continue
        shutil.copy(os.path.join(ep, "motion_50hz.csv"), os.path.join(csv_dir, os.path.basename(ep) + ".csv"))
        kept += 1
        total_s += meta["duration_sim_s"]
    print(f"{kept} episodes, {total_s / 60:.1f} min of motion -> {csv_dir}")
    if a.no_convert or kept == 0:
        return
    out_pkl = os.path.join(os.path.abspath(a.out), "locomanip_motion_lib.pkl")
    conv = os.path.join(SONIC_ROOT, "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py")
    cmd = [a.sonic_python, conv, "--input", os.path.abspath(csv_dir), "--output", out_pkl, "--fps", "50"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=SONIC_ROOT)


if __name__ == "__main__":
    main()
