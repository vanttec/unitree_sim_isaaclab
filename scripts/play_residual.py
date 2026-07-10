"""Evaluate a trained residual-grasp policy: success rate under jitter vs the
base-only 30% baseline.

    isaac-python scripts/play_residual.py \
        --task Isaac-ResidualGrasp-TennisBall-State-G129-Inspire-Joint \
        --num_envs 256 --episodes 4 --headless

Each parallel env is one jittered placement; lift >= --lift_thresh counts as a
grasp success. Reports the aggregate success rate.
"""
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-ResidualGrasp-TennisBall-State-G129-Inspire-Joint")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--lift_thresh", type=float, default=0.05)
parser.add_argument("--checkpoint", type=str, default=None, help="explicit .pt; else latest run")
parser.add_argument("--base_only", action="store_true", help="zero the residual (baseline)")
parser.add_argument("--cartesian_ff", action="store_true", help="hardcoded IK jitter correction (no policy) - ceiling test")
parser.add_argument("--cartesian_gain", type=float, default=1.0)
parser.add_argument("--no_jitter", action="store_true", help="disable object placement jitter (measure the ceiling)")
parser.add_argument("--fixed_jitter", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="place object at a FIXED offset (m) in every env, e.g. --fixed_jitter 0.03 0.03 (extreme corner)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

import tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
# eval is a fixed test: no curriculum (would start easy at +-1cm and inflate it)
env_cfg.curriculum = None
if args_cli.fixed_jitter is not None:
    # deterministic extreme placement in every env -> watch if the ball rolls off
    fx, fy = args_cli.fixed_jitter
    env_cfg.events.reset_object.params["pose_range"] = {"x": (fx, fx), "y": (fy, fy)}
    print(f"[play] FIXED jitter offset x={fx} y={fy}")
elif args_cli.no_jitter:
    # zero the placement jitter -> measures the base grasp's ceiling (does a
    # perfectly-positioned grasp hold through the lift, or grab-and-drop?)
    env_cfg.events.reset_object.params["pose_range"] = {}
else:
    # pin the FULL jitter the residual must handle (curriculum only applies in train)
    env_cfg.events.reset_object.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg.device = args_cli.device

if args_cli.cartesian_ff:
    env_cfg.actions.residual.cartesian_ff = True
    env_cfg.actions.residual.cartesian_gain = args_cli.cartesian_gain

env = gym.make(args_cli.task, cfg=env_cfg)
if args_cli.base_only:
    env.unwrapped.action_manager.get_term("residual")._residual_scale = 0.0
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

no_policy = args_cli.base_only or args_cli.cartesian_ff
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
if not no_policy:
    ckpt = args_cli.checkpoint or get_checkpoint_path(
        os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name),
        run_dir=".*", other_dirs=None, checkpoint="model_.*.pt",
    )
    print(f"[play] loading {ckpt}")
    runner.load(ckpt)
policy = runner.get_inference_policy(device=args_cli.device)

obj = env.unwrapped.scene["object"]
robot = env.unwrapped.scene["robot"]
_bn = list(robot.data.body_names)
_gi = next(i for i, n in enumerate(_bn) if "R_index_proximal" in n)
ep_len = int(env.unwrapped.max_episode_length)
n_env = args_cli.num_envs
successes = 0
total = 0
HOLD_STEPS = 100  # ~1 s of stable held-aloft grasp = success (robust to transients)
_term = env.unwrapped.action_manager.get_term("residual")
with torch.inference_mode():
    for ep in range(args_cli.episodes):
        env.reset()
        obs = env.get_observations()
        z0 = obj.data.root_pos_w[:, 2].clone()
        xy0 = obj.data.root_pos_w[:, :2].clone()   # jittered ball xy
        zmax = z0.clone()
        held_count = torch.zeros(n_env, device=args_cli.device)
        _dbg_done = False
        # stop 2 steps short of max_episode_length so the terminal auto-reset
        # never fires inside the loop and corrupts the measurement
        for _ in range(ep_len - 2):
            actions = policy(obs) if not no_policy else torch.zeros(
                (n_env, env.num_actions), device=args_cli.device)
            obs, _, _, _ = env.step(actions)
            # after the arm residual has latched, dump it vs the ball offset to
            # verify the residual RESPONDS to the jitter (should flip with x/y sign)
            if ep == 0 and not _dbg_done and float(_term.phase_fraction[0]) > _term._freeze_phase + 0.02:
                off = (obj.data.root_pos_w[:, :2] - torch.tensor([-0.15, 0.40], device=args_cli.device))
                for e in range(min(n_env, 4)):
                    fr = _term._frozen[e]
                    print(f"[dbg] env{e}  ball_off=({off[e,0]:+.3f},{off[e,1]:+.3f})  "
                          f"|Δq|={fr.norm():.4f}  Δq={[round(float(v),3) for v in fr]}")
                _dbg_done = True
            z = obj.data.root_pos_w[:, 2]
            zmax = torch.maximum(zmax, z)
            dist = torch.norm(robot.data.body_pos_w[:, _gi] - obj.data.root_pos_w, dim=-1)
            speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
            held_now = ((z - z0) >= args_cli.lift_thresh) & (dist < 0.09) & (speed < 0.5)
            held_count += held_now.float()
        ok = held_count >= HOLD_STEPS
        successes += int(ok.sum().item())
        total += n_env
        print(f"[play] ep {ep}  held {int(ok.sum())}/{n_env}  "
              f"peak_lift={float((zmax - z0).max()):+.2f}  mean_held_steps={float(held_count.mean()):.0f}")

_mode = "CARTESIAN-FF" if args_cli.cartesian_ff else ("BASE-ONLY" if args_cli.base_only else "RESIDUAL")
print(f"\n[play] {_mode}  "
      f"success {successes}/{total} = {successes/total:.0%}  (baseline was ~30%)\n")

env.close()
app.close()
