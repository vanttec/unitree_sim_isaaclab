# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""rsl_rl PPO config for the residual-grasp task (asymmetric actor-critic).

Actor sees the "policy" group (deployable: arm proprio + phase + object pose /
later pixels). Critic sees the "critic" group (privileged object pose). The
action is a small bounded arm residual, so networks stay small.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ResidualGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500   # tuned for ~512 envs (24*512=12k transitions/iter);
                           # converges ~100-200 iters. Raise if using fewer envs.
    save_interval = 50
    experiment_name = "residual_grasp_tennisball"
    run_name = ""

    # actor <- "policy" group, critic <- "critic" group (asymmetric)
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        # tiny initial noise: the base grasp is precarious — big exploration
        # (tanh saturates ~0.7) swings the full residual every step and knocks
        # the ball loose before grasp, so the lift reward is never sampled. Keep
        # noise small so the residual explores gently and the grasp survives.
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,   # was 0.005 — it pumped noise up (stuck 0.72); let it shrink
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
