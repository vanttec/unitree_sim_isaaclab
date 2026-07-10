# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Residual single-arm action for robust semantic grasps.

The command sent to the robot each control step is::

    q_cmd(t) = q_base(t) + Δq_θ(o_t)

``q_base`` is a recorded semantic-grasp trajectory (``grasp_bridge`` slot: the
BCI-decoded grasp shape) replayed by phase. The policy only outputs ``Δq_θ`` — a
small, bounded correction applied **exclusively to the working arm** (default:
right, 7 DOF). Every other joint — the whole hand — is driven byte-exact from the
recording, so the learned residual can move the hand *to* the object but can
never reshape *what the hand does*. Grasp identity is preserved by construction.

Residual bound: ``Δq = residual_scale * tanh(raw_action)`` → stays near the base
trajectory (sim2real error rides only the small correction) and cannot wander
into a wildly different arm configuration.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from grasp_bridge.inspire_convert import _DDS_RANGES, DDS_INSPIRE_JOINT_NAMES, dds_norm_to_rad
from grasp_bridge.trajectory_io import load_trajectory

# DDS 29-DOF body motor order (matches grasp_bridge.g1_constants indices).
_DDS_BODY_JOINT_NAMES: tuple[str, ...] = (
    # left leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm (22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = _DDS_BODY_JOINT_NAMES[22:29]
_LEFT_ARM_JOINT_NAMES: tuple[str, ...] = _DDS_BODY_JOINT_NAMES[15:22]

# Inspire intermediate/distal joints are coupled to a proximal DOF, exactly as the
# teleop path does (action_provider_dds.special_joint_mapping): the driven target is
# ``proximal_rad[dds_source] * scale``. Must be replicated or replayed fingertips
# stay straight and the grasp shape is wrong.  {joint_name: (dds_source_index, scale)}
_INSPIRE_COUPLING: dict[str, tuple[int, float]] = {
    "L_index_intermediate_joint": (9, 1.0),
    "L_middle_intermediate_joint": (8, 1.0),
    "L_pinky_intermediate_joint": (6, 1.0),
    "L_ring_intermediate_joint": (7, 1.0),
    "L_thumb_intermediate_joint": (10, 1.5),
    "L_thumb_distal_joint": (10, 2.4),
    "R_index_intermediate_joint": (3, 1.0),
    "R_middle_intermediate_joint": (2, 1.0),
    "R_pinky_intermediate_joint": (0, 1.0),
    "R_ring_intermediate_joint": (1, 1.0),
    "R_thumb_intermediate_joint": (4, 1.5),
    "R_thumb_distal_joint": (4, 2.4),
}


class ResidualArmAction(ActionTerm):
    """Replay a base grasp trajectory, add a bounded residual on one arm."""

    cfg: ResidualArmActionCfg
    _asset: Articulation

    def __init__(self, cfg: ResidualArmActionCfg, env) -> None:
        super().__init__(cfg, env)

        # --- resolve residual (arm) joint ids by name -----------------------
        arm_names = (
            _RIGHT_ARM_JOINT_NAMES if cfg.arm == "right" else _LEFT_ARM_JOINT_NAMES
        )
        self._arm_ids, _ = self._asset.find_joints(list(arm_names), preserve_order=True)
        self._arm_ids = torch.tensor(self._arm_ids, device=self.device, dtype=torch.long)
        self._num_arm = len(arm_names)

        # --- Cartesian modes (share the Jacobian setup) -----------------------
        # cartesian_ff: hardcoded IK shift-by-jitter (test, no policy).
        # cartesian_action: policy outputs a 3D wrist offset -> IK -> Δq. The map
        #   jitter->wrist-shift is ~identity, so RL learns it (unlike joint-space
        #   which is inverse kinematics and collapsed to a constant).
        self._cart_ff = bool(cfg.cartesian_ff)
        self._cart_action = bool(cfg.cartesian_action)
        self._cart_ff_base = bool(cfg.cartesian_ff_base)
        if self._cart_ff or self._cart_action:
            ee_ids, _ = self._asset.find_bodies(cfg.ee_body_name, preserve_order=True)
            self._ee_body_idx = int(ee_ids[0])
            # grasp-center point: land the FINGERS on the ball, not the wrist. The
            # wrist Jacobian is translated to this point (better analytic base).
            gp_ids, _ = self._asset.find_bodies("R_index_proximal", preserve_order=True)
            self._grasp_body_idx = int(gp_ids[0]) if len(gp_ids) else self._ee_body_idx
            self._cart_gain = float(cfg.cartesian_gain)
            self._cart_action_scale = float(cfg.cartesian_action_scale)
            self._cart_dbg = True

        # --- residual scale (per-joint or scalar) ---------------------------
        scale = cfg.residual_scale
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * self._num_arm
        assert len(scale) == self._num_arm, "residual_scale must be scalar or len == arm DOF"
        self._residual_scale = torch.tensor(scale, device=self.device, dtype=torch.float32)

        # residual is live during approach, then latched constant from freeze_phase
        self._freeze_phase = float(cfg.freeze_phase)
        self._frozen = torch.zeros(self.num_envs, self._num_arm, device=self.device)

        # --- optional bounded HAND residual (adapt grip to the offset ball) ----
        # 6 driven right-hand DOF (DDS 0..5); drives coupled intermediate/distal
        # joints too so the finger shape stays coherent = grasp TYPE preserved.
        self._hand_scale = float(cfg.hand_residual_scale)
        self._num_hand = 6 if self._hand_scale > 0.0 else 0
        if self._num_hand:
            names = self._asset.joint_names
            n2c = {n: i for i, n in enumerate(names)}
            num_joints = len(names)
            # M: (6, num_joints) maps a right-hand DDS residual to joint deltas
            M = torch.zeros(6, num_joints, device=self.device)
            for dds_i in range(6):                       # driven joints
                col = n2c.get(DDS_INSPIRE_JOINT_NAMES[dds_i])
                if col is not None:
                    M[dds_i, col] = 1.0
            for nm, (src, sc) in _INSPIRE_COUPLING.items():   # coupled -> follow proximal
                if nm.startswith("R_") and src < 6:
                    col = n2c.get(nm)
                    if col is not None:
                        M[src, col] = sc
            self._hand_M = M
            self._hand_freeze_phase = float(cfg.hand_freeze_phase)
            self._hand_frozen = torch.zeros(self.num_envs, self._num_hand, device=self.device)

        # --- build the base-pose buffer (T, num_joints) in env joint order --
        self._base = self._build_base_buffer()          # (T, num_joints)
        self._num_frames = self._base.shape[0]

        # --- phase bookkeeping ----------------------------------------------
        # recorded at playback_hz; advance phase by frames-per-control-step.
        self._frames_per_step = float(cfg.playback_hz) * float(env.step_dt)
        self._phase = torch.zeros(self.num_envs, device=self.device)   # float frame index

        # --- action buffers -------------------------------------------------
        self._act_dim = 3 if self._cart_action else (self._num_arm + self._num_hand)
        self._raw_actions = torch.zeros(self.num_envs, self._act_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    # ------------------------------------------------------------------ props
    @property
    def action_dim(self) -> int:
        return self._act_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def phase_fraction(self) -> torch.Tensor:
        """Normalized trajectory phase in [0, 1], shape (num_envs,)."""
        return (self._phase / max(1, self._num_frames - 1)).clamp(0.0, 1.0)

    @property
    def arm_joint_ids(self) -> torch.Tensor:
        return self._arm_ids

    # ----------------------------------------------------------------- update
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self._cart_action:
            # bounded 3D wrist offset (m): Δx = scale * tanh(a)
            self._processed_actions[:] = self._cart_action_scale * torch.tanh(actions)
            return
        # bounded residual: Δq = scale * tanh(a). arm cols then hand cols.
        arm = actions[:, : self._num_arm]
        self._processed_actions[:, : self._num_arm] = self._residual_scale * torch.tanh(arm)
        if self._num_hand:
            hand = actions[:, self._num_arm :]
            self._processed_actions[:, self._num_arm :] = self._hand_scale * torch.tanh(hand)

    def apply_actions(self) -> None:
        # advance phase, clamp to hold last frame at the end
        self._phase += self._frames_per_step
        idx = self._phase.clamp(max=self._num_frames - 1).long()   # (num_envs,)

        target = self._base[idx].clone()                           # (num_envs, num_joints)

        # Cartesian modes: convert a wrist offset Δx -> joint delta Δq via IK,
        # one-shot latched (constant spatial shift, no jiggle).
        if self._cart_ff or self._cart_action:
            if self._cart_ff:
                dx = self._cart_gain * self._jitter_offset()      # hardcoded test
            else:
                dx = self._processed_actions                       # policy 3D residual
                if self._cart_ff_base:
                    # RPL: analytic IK base (shift by observed jitter) + learned
                    # residual on top. Policy starts at ~82% and only improves.
                    dx = dx + self._cart_gain * self._jitter_offset()
            dq = self._ik_from_dx(dx)
            before = (self.phase_fraction < self._freeze_phase).unsqueeze(-1)
            self._frozen = torch.where(before, dq, self._frozen)
            target[:, self._arm_ids] += torch.where(before, dq, self._frozen)
            self._asset.set_joint_position_target(target)
            return

        # ARM residual: one-shot latch at freeze_phase -> constant spatial shift,
        # no per-step jiggle during the hold.
        arm_res = self._processed_actions[:, : self._num_arm]
        before = (self.phase_fraction < self._freeze_phase).unsqueeze(-1)
        self._frozen = torch.where(before, arm_res, self._frozen)
        target[:, self._arm_ids] += torch.where(before, arm_res, self._frozen)

        # HAND residual: live THROUGH the grasp-close (fingers conform to the
        # offset ball), then latched so the adapted grip holds without jiggle.
        # Drives coupled finger joints via _hand_M so the grasp SHAPE stays intact.
        if self._num_hand:
            hand_res = self._processed_actions[:, self._num_arm :]
            hbefore = (self.phase_fraction < self._hand_freeze_phase).unsqueeze(-1)
            self._hand_frozen = torch.where(hbefore, hand_res, self._hand_frozen)
            happlied = torch.where(hbefore, hand_res, self._hand_frozen)
            target = target + happlied @ self._hand_M              # (n,6)@(6,J)=(n,J)

        self._asset.set_joint_position_target(target)

    def _jitter_offset(self) -> torch.Tensor:
        """Observed object displacement from nominal spawn (the jitter), (n,3)."""
        obj = self._env.scene["object"]
        origins = self._env.scene.env_origins
        nominal = torch.tensor((-0.15, 0.40, 0.87535), device=self.device)
        return (obj.data.root_pos_w - origins) - nominal

    def _ik_from_dx(self, dx: torch.Tensor) -> torch.Tensor:
        """Damped-least-squares IK: joint delta realizing wrist offset dx, (n,num_arm)."""
        jac = self._asset.root_physx_view.get_jacobians()      # (n, nb, 6, ndof)
        ji = self._ee_body_idx - 1                             # fixed-base: drop base row
        J6 = jac[:, ji, :, :][:, :, self._arm_ids]             # (n,6,num_arm) full 6D
        Jv, Jw = J6[:, :3, :], J6[:, 3:, :]
        # translate reference wrist-origin -> grasp-center point:
        # v_P = v_O + w x r  =>  Jv_P = Jv - cross(r, Jw),  r = grasp - wrist (world)
        bp = self._asset.data.body_pos_w
        r = bp[:, self._grasp_body_idx] - bp[:, self._ee_body_idx]          # (n,3)
        J = Jv - torch.cross(r.unsqueeze(-1).expand(-1, -1, Jw.shape[-1]), Jw, dim=1)
        lam = 0.25   # strong damping: near-singular configs during LIVE grasp blew
                     # up dq to ~1 rad at lam=0.15 -> arm flails, ball flung
        JT = J.transpose(1, 2)                                 # (n,num_arm,3)
        A = J @ JT + (lam ** 2) * torch.eye(3, device=self.device)
        dq = (JT @ torch.linalg.solve(A, dx.unsqueeze(-1))).squeeze(-1)
        # hard cap: singularity can still amplify -> clamp per-env dq norm so a
        # bad Jacobian can never yank the arm (base stays in control)
        _MAXQ = 0.15
        n = dq.norm(dim=-1, keepdim=True)
        dq = dq * (_MAXQ / n.clamp_min(_MAXQ))
        if self._cart_dbg:
            print(f"[cart] jac{tuple(jac.shape)} J{tuple(J.shape)} "
                  f"dx0={[round(float(v),3) for v in dx[0]]} "
                  f"dq0={[round(float(v),3) for v in dq[0]]}")
            self._cart_dbg = False
        return dq

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._phase.zero_()
            self._frozen.zero_()
            if self._num_hand:
                self._hand_frozen.zero_()
        else:
            self._phase[env_ids] = 0.0
            self._frozen[env_ids] = 0.0
            if self._num_hand:
                self._hand_frozen[env_ids] = 0.0

    # ------------------------------------------------------------- base build
    def _build_base_buffer(self) -> torch.Tensor:
        """Map recorded body_cmd (DDS-29) + hand_q (Inspire-12) to env joint order.

        Joints the recording does not name (e.g. hand mimic/distal DOF) default to
        the articulation's default joint position, so every DOF has a sane target.
        """
        traj = load_trajectory(self.cfg.slot)
        body_cmd = np.asarray(traj["body_cmd"], dtype=np.float32)   # (T, 29) rad, DDS order
        hand_norm = np.asarray(traj["hand_q"], dtype=np.float32)    # (T, 12) DDS normalized
        n_frames = body_cmd.shape[0]

        env_names: list[str] = self._asset.joint_names
        num_joints = len(env_names)
        name_to_col = {name: i for i, name in enumerate(env_names)}

        # start every joint at its default target, then overwrite recorded ones
        default = self._asset.data.default_joint_pos[0].detach().cpu().numpy()  # (num_joints,)
        base = np.broadcast_to(default, (n_frames, num_joints)).copy()

        # body (29) by name
        for dds_i, name in enumerate(_DDS_BODY_JOINT_NAMES):
            col = name_to_col.get(name)
            if col is not None:
                base[:, col] = body_cmd[:, dds_i]

        # hand: DDS normalized (12) -> radians per DDS index (T, 12)
        hand_rad = np.empty_like(hand_norm)
        for dds_i in range(12):
            hand_rad[:, dds_i] = [dds_norm_to_rad(float(v), dds_i) for v in hand_norm[:, dds_i]]

        # grip boost: over-close the fingers past contact so a standing position
        # error remains -> normal force N = Kp*error > 0 -> friction can hold the
        # object. Scaled by how closed the finger already is (0 when open), clamped
        # to the joint's closed limit. Needed only when NOT using kinematic attach.
        boost = float(self.cfg.hand_close_boost)
        if boost > 0.0:
            hi = np.array([_DDS_RANGES[i][1] for i in range(12)], dtype=np.float32)
            closure_frac = np.clip(hand_rad / hi, 0.0, 1.0)
            hand_rad = np.minimum(hand_rad + boost * closure_frac, hi)

        # 12 proximal/yaw joints driven directly
        for dds_i, name in enumerate(DDS_INSPIRE_JOINT_NAMES):
            col = name_to_col.get(name)
            if col is not None:
                base[:, col] = hand_rad[:, dds_i]

        # 12 intermediate/distal joints coupled to a proximal (matches teleop)
        for name, (src_dds_i, scale) in _INSPIRE_COUPLING.items():
            col = name_to_col.get(name)
            if col is not None:
                base[:, col] = hand_rad[:, src_dds_i] * scale

        return torch.tensor(base, device=self.device, dtype=torch.float32)


@configclass
class ResidualArmActionCfg(ActionTermCfg):
    """Config for :class:`ResidualArmAction`."""

    class_type: type[ActionTerm] = ResidualArmAction

    slot: int = MISSING
    """grasp_bridge trajectory slot to replay as the base (1=coin ... 5=container)."""

    arm: str = "right"
    """which arm the residual acts on ("right" or "left")."""

    residual_scale: float | list[float] = 0.08
    """max residual magnitude per arm joint, rad. Δq = residual_scale * tanh(a).
    Size to expected placement jitter (a few cm ≈ 0.05-0.10 rad at the wrist)."""

    playback_hz: float = 100.0
    """rate the base trajectory was recorded at (grasp_bridge default = 100 Hz)."""

    hand_close_boost: float = 0.0
    """over-close the fingers past the recorded target, rad, to build grip normal
    force for real (non-attach) physical grasping. 0 = replay exactly. Try ~0.2-0.4
    when kinematic attach is off; leave 0 when attach handles holding."""

    freeze_phase: float = 0.40
    """phase fraction at which the residual is latched and held constant. Live
    (policy-driven) while phase<freeze_phase (approach), frozen after — so the
    approach-alignment correction persists through the grasp+lift without any
    step-to-step jiggle that would pop the grasp loose. Set just as the fingers
    begin to close (~0.38-0.40 for the tennisball trajectory)."""

    hand_residual_scale: float = 0.0
    """max bounded residual on the 6 driven right-hand DOF, rad. 0 = hand frozen
    (grasp byte-exact). >0 lets the policy adapt the grip to conform to an
    off-center jittered ball; kept small so the grasp TYPE (BCI identity) is
    preserved. Coupled finger joints follow so the finger shape stays coherent."""

    hand_freeze_phase: float = 0.55
    """phase fraction at which the hand residual latches. Live through the grasp
    close (fingers conform to the ball), frozen after so the adapted grip holds
    steady. Set just after the fingers finish closing."""

    cartesian_action: bool = False
    """policy outputs a 3D wrist-position offset (m) instead of 7 joint residuals;
    converted to Δq via damped-LS IK. The jitter->wrist-offset map is ~identity,
    so RL learns it (joint-space required inverse kinematics and collapsed to a
    constant). One-shot latched at freeze_phase."""

    cartesian_action_scale: float = 0.06
    """max wrist offset the Cartesian policy can command, m (bounds ~+-6cm)."""

    cartesian_ff_base: bool = False
    """RPL: use the analytic IK jitter correction as the BASE and add the policy's
    Cartesian output as a residual on top. Policy starts at the analytic ceiling
    (~82%) and learns only the last-mile correction. Requires object pose (sim /
    privileged) — for vision the base becomes a perception-estimated correction."""

    cartesian_ff: bool = False
    """TEST mode: ignore the policy and hardcode an IK correction that shifts the
    wrist by the observed object jitter. Validates whether position-conditioning
    recovers the jitter loss (ceiling) before investing in a learned Cartesian
    residual."""

    cartesian_gain: float = 1.0
    """scale on the jitter->wrist-shift feed-forward (1.0 = shift wrist exactly by
    the observed jitter)."""

    ee_body_name: str = "right_hand_base_link"
    """body used as the wrist for the Cartesian Jacobian."""
