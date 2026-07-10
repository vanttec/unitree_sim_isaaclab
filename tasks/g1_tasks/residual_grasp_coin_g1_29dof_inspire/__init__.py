# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import residual_grasp_coin_env_cfg

gym.register(
    id="Isaac-ResidualGrasp-Coin-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": residual_grasp_coin_env_cfg.ResidualGraspCoinEnvCfg,
    },
    disable_env_checker=True,
)
