# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Curriculum: grow the object placement jitter from easy -> full over training.

The residual has to learn object_pose -> arm-correction. Under the full +-3cm
jitter the large offsets almost never grasp during exploration, so there's no
gradient to learn their correction (policy plateaus ~= base). Starting at +-1cm
lets it learn the centering mapping on cases it CAN grasp, then grow the range so
that mapping transfers to the hard offsets.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_jitter_range(
    env: "ManagerBasedRLEnv",
    env_ids,
    start_mag: float = 0.01,
    end_mag: float = 0.03,
    num_steps: int = 8000,
    term_name: str = "reset_object",
) -> float:
    """Linearly grow reset_object x/y jitter start_mag -> end_mag over num_steps
    control steps. Returns the current magnitude (logged by the curriculum mgr)."""
    frac = min(env.common_step_counter / max(num_steps, 1), 1.0)
    mag = start_mag + frac * (end_mag - start_mag)
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["pose_range"] = {"x": (-mag, mag), "y": (-mag, mag)}
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return mag
