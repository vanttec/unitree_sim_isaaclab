# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Reset event: teleport the robot to the residual base-trajectory's FRAME 0.

Without this, the robot resets to its *default* pose while playback starts at the
recorded trajectory's frame 0 — a different pose. The PD controller then yanks the
arm default->frame0 over the first steps, sweeping the hand THROUGH the object and
toppling it before the grasp begins (the reset de-sync). Teleporting the joint
*state* (not just the target) to frame 0 removes the sweep, so the recorded grasp
replays faithfully — exactly as it did during teleop capture.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_to_trajectory_start(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    action_name: str = "residual",
) -> None:
    term = env.action_manager.get_term(action_name)
    robot = term._asset                       # articulation the residual drives
    base0 = term._base[0]                      # (num_joints,) frame-0 target, env order

    n = len(env_ids)
    pos = base0.unsqueeze(0).expand(n, -1).contiguous()
    vel = torch.zeros_like(pos)

    # teleport joint state (no PD sweep) and hold the pose so it doesn't drift
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
    robot.set_joint_position_target(pos, env_ids=env_ids)
