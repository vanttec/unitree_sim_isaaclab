"""Teleoperated loco-manipulation recording (sim ground truth) for SONIC fine-tuning.

recorder.py  in-process recorder hooked into sim_main.py (--locomanip_record)
cli.py       start / stop / discard episodes over DDS (rt/locomanip/record_cmd)
export.py    episodes -> Bones-SEED CSVs -> SONIC motion_lib PKL
"""
