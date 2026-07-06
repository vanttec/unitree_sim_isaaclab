"""Attach pick-place objects to Inspire hands when grasp is firm; release when opening."""

from __future__ import annotations

from typing import Optional

import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


class ObjectGraspController:
    """Keep graspable rigid objects fixed on the pedestal until a firm hand grasp."""

    def __init__(
        self,
        env,
        inspire_dds,
        *,
        grasp_distance: float = 0.08,
        grasp_closure: float = 0.58,
        release_closure: float = 0.28,
        attach_hold_frames: int = 12,
        enabled: bool = True,
    ):
        self.env = env
        self.inspire_dds = inspire_dds
        self.grasp_distance = grasp_distance
        self.grasp_closure = grasp_closure
        self.release_closure = release_closure
        self.attach_hold_frames = attach_hold_frames
        self.enabled = enabled

        self.robot = env.scene["robot"]
        self.object = env.scene["object"]
        self.device = env.device

        names = list(self.robot.data.body_names)
        self._left_wrist_idx = self._find_body(names, ("left_wrist_yaw", "left_wrist"))
        self._right_wrist_idx = self._find_body(names, ("right_wrist_yaw", "right_wrist"))
        self._left_grasp_idx = self._find_body_optional(names, ("L_index_proximal", "left_hand_index", "left_index"))
        self._right_grasp_idx = self._find_body_optional(names, ("R_index_proximal", "right_hand_index", "right_index"))

        self._attached_side: Optional[str] = None
        self._offset_pos: Optional[torch.Tensor] = None
        self._offset_quat: Optional[torch.Tensor] = None
        self._hand_cmd_norm = [0.0] * 12
        self._left_attach_streak = 0
        self._right_attach_streak = 0

        if inspire_dds is not None:
            inspire_dds.register_hand_cmd_callback(self._on_hand_cmd)

    @staticmethod
    def _find_body(names: list[str], candidates: tuple[str, ...]) -> int:
        for token in candidates:
            for i, name in enumerate(names):
                if token in name:
                    return i
        raise ValueError(f"Could not find wrist body in {names}")

    @staticmethod
    def _find_body_optional(names: list[str], candidates: tuple[str, ...]) -> int | None:
        try:
            return ObjectGraspController._find_body(names, candidates)
        except ValueError:
            return None

    def _on_hand_cmd(self, norm_q: list[float]) -> None:
        self._hand_cmd_norm = norm_q[:12]

    def reset(self) -> None:
        self._attached_side = None
        self._offset_pos = None
        self._offset_quat = None
        self._left_attach_streak = 0
        self._right_attach_streak = 0

    def _hand_motors(self, side: str) -> tuple[float, float, float, float, float, float]:
        """DDS order per hand: pinky, ring, middle, index, thumb_pitch, thumb_yaw."""
        base = 6 if side == "left" else 0
        q = self._hand_cmd_norm[base : base + 6]
        if len(q) < 6:
            return (0.0,) * 6
        return tuple(float(v) for v in q)

    def _closure(self, side: str) -> float:
        pinky, ring, middle, index, thumb_p, _thumb_y = self._hand_motors(side)
        power_grasp = (pinky + ring + middle + index) / 4.0
        pinch_grasp = (index + thumb_p) / 2.0
        return max(power_grasp, pinch_grasp)

    def _is_released(self, side: str) -> bool:
        pinky, ring, middle, index, thumb_p, _thumb_y = self._hand_motors(side)
        pinch_held = (index + thumb_p) / 2.0 > self.release_closure
        power_held = (pinky + ring + middle + index) / 4.0 > self.release_closure
        return not pinch_held and not power_held

    def _near_and_closing(self, side: str, dist: float, closure: float) -> bool:
        return dist <= self.grasp_distance and closure >= self.grasp_closure

    def _grasp_point(self, side: str) -> torch.Tensor:
        if side == "left":
            idx = self._left_grasp_idx if self._left_grasp_idx is not None else self._left_wrist_idx
        else:
            idx = self._right_grasp_idx if self._right_grasp_idx is not None else self._right_wrist_idx
        return self.robot.data.body_pos_w[0, idx]

    def _wrist_pose(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.robot.data.body_pos_w[0, idx]
        quat = self.robot.data.body_quat_w[0, idx]
        return pos, quat

    def _try_attach(self, side: str, wrist_idx: int) -> None:
        obj_pos = self.object.data.root_pos_w[0]
        obj_quat = self.object.data.root_quat_w[0]
        wrist_pos, wrist_quat = self._wrist_pose(wrist_idx)
        rel_pos, rel_quat = subtract_frame_transforms(wrist_pos, wrist_quat, obj_pos, obj_quat)
        self._attached_side = side
        self._offset_pos = rel_pos
        self._offset_quat = rel_quat

    def _follow_wrist(self, wrist_idx: int) -> None:
        wrist_pos, wrist_quat = self._wrist_pose(wrist_idx)
        new_pos, new_quat = combine_frame_transforms(
            wrist_pos, wrist_quat, self._offset_pos, self._offset_quat
        )
        root_pose = torch.cat([new_pos, new_quat]).unsqueeze(0)
        zero_vel = torch.zeros((1, 6), device=self.device)
        self.object.write_root_pose_to_sim(root_pose)
        self.object.write_root_velocity_to_sim(zero_vel)

    def update(self) -> None:
        if not self.enabled or self.object is None:
            return

        obj_pos = self.object.data.root_pos_w[0]
        d_left = torch.linalg.norm(obj_pos - self._grasp_point("left")).item()
        d_right = torch.linalg.norm(obj_pos - self._grasp_point("right")).item()

        c_left = self._closure("left")
        c_right = self._closure("right")

        if self._attached_side is None:
            if self._near_and_closing("left", d_left, c_left):
                self._left_attach_streak += 1
            else:
                self._left_attach_streak = 0
            if self._near_and_closing("right", d_right, c_right):
                self._right_attach_streak += 1
            else:
                self._right_attach_streak = 0

            if (
                self._left_attach_streak >= self.attach_hold_frames
                and c_left >= c_right
            ):
                self._try_attach("left", self._left_wrist_idx)
                self._left_attach_streak = 0
            elif self._right_attach_streak >= self.attach_hold_frames:
                self._try_attach("right", self._right_wrist_idx)
                self._right_attach_streak = 0
        else:
            side = self._attached_side
            wrist_idx = self._left_wrist_idx if side == "left" else self._right_wrist_idx
            if self._is_released(side):
                self.reset()
                return
            self._follow_wrist(wrist_idx)
