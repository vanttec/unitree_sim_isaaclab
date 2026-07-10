# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import residual_grasp_tennisball_env_cfg
from . import residual_grasp_tennisball_state_env_cfg
from .agents import rsl_rl_ppo_cfg

gym.register(
    id="Isaac-ResidualGrasp-TennisBall-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": residual_grasp_tennisball_env_cfg.ResidualGraspTennisBallEnvCfg,
    },
    disable_env_checker=True,
)

# state-based variant (no cameras) — avoids IsaacLab #3312, for RL bring-up
gym.register(
    id="Isaac-ResidualGrasp-TennisBall-State-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": residual_grasp_tennisball_state_env_cfg.ResidualGraspTennisBallStateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:ResidualGraspPPORunnerCfg",
    },
    disable_env_checker=True,
)
