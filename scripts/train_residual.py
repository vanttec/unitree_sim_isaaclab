"""Train the residual-grasp policy with rsl_rl PPO (asymmetric actor-critic).

State-based bring-up (no cameras -> avoids IsaacLab #3312):
    isaac-python scripts/train_residual.py \
        --task Isaac-ResidualGrasp-TennisBall-State-G129-Inspire-Joint \
        --num_envs 256 --headless

Vision later: same command with the vision task id once #3312 is resolved.
"""
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-ResidualGrasp-TennisBall-State-G129-Inspire-Joint")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from datetime import datetime

import tasks  # noqa: F401  (registers the envs)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# configs
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg.device = args_cli.device
agent_cfg.seed = args_cli.seed
if args_cli.max_iterations is not None:
    agent_cfg.max_iterations = args_cli.max_iterations
agent_cfg.resume = args_cli.resume

# log dir
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name, stamp)
os.makedirs(log_dir, exist_ok=True)
print(f"[train] task={args_cli.task}  envs={args_cli.num_envs}  log={log_dir}")

# env -> rsl_rl
env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

env.close()
app.close()
