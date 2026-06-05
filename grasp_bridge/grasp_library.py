"""Hardcoded grasp definitions — tune joint values in sim."""

from dataclasses import dataclass

from .g1_constants import INSPIRE_OPEN


@dataclass(frozen=True)
class GraspKeyframe:
    """One snapshot: 14 arm joints (L7 + R7) + 12 Inspire motor commands (normalized q)."""

    name: str
    duration_s: float
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    hand_q: tuple[float, ...]


@dataclass(frozen=True)
class GraspSequence:
    grasp_id: int
    label: str
    keyframes: tuple[GraspKeyframe, ...]
    active_side: str = "left"  # "left" | "right" — el otro brazo se mantiene quieto


def _hand(
    r_pinky=0.0,
    r_ring=0.0,
    r_middle=0.0,
    r_index=0.0,
    r_thumb_p=0.0,
    r_thumb_y=0.0,
    l_pinky=0.0,
    l_ring=0.0,
    l_middle=0.0,
    l_index=0.0,
    l_thumb_p=0.0,
    l_thumb_y=0.0,
) -> tuple[float, ...]:
    """DDS order: right hand [0-5], left hand [6-11]."""
    return (
        r_pinky, r_ring, r_middle, r_index, r_thumb_p, r_thumb_y,
        l_pinky, l_ring, l_middle, l_index, l_thumb_p, l_thumb_y,
    )


def _hand_left(**kwargs) -> tuple[float, ...]:
    """Solo mano izquierda; derecha abierta."""
    left = {f"l_{k}": v for k, v in kwargs.items()}
    return _hand(**left)


# 7 joints: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
_APPROACH_LEFT = (0.35, 0.25, -0.15, 0.85, 0.0, 0.2, 0.0)
_LIFT_LEFT = (0.25, 0.25, -0.15, 0.90, 0.0, 0.2, 0.0)
_APPROACH_RIGHT = (-0.35, -0.25, 0.15, 0.85, 0.0, -0.2, 0.0)
_LIFT_RIGHT = (-0.25, -0.25, 0.15, 0.75, 0.0, -0.2, 0.0)


def _sequence_left(grasp_id: int, label: str, close_hand: tuple[float, ...]) -> GraspSequence:
    """Solo brazo y mano izquierdos."""
    return GraspSequence(
        grasp_id=grasp_id,
        label=label,
        active_side="left",
        keyframes=(
            GraspKeyframe("approach", 5.0, _APPROACH_LEFT, _APPROACH_LEFT, INSPIRE_OPEN),
            GraspKeyframe("close", 5.0, _APPROACH_LEFT, _APPROACH_LEFT, close_hand),
            GraspKeyframe("lift", 5.0, _LIFT_LEFT, _LIFT_LEFT, close_hand),
        ),
    )


GRASPS: dict[int, GraspSequence] = {
    1: _sequence_left(1, "power", _hand_left(pinky=0.85, ring=0.85, middle=0.85, index=0.85, thumb_p=0.8, thumb_y=0.5)),
    2: _sequence_left(2, "pinch", _hand_left(index=0.92, thumb_p=0.9, thumb_y=0.4)),
    3: _sequence_left(3, "lateral", _hand_left(thumb_p=0.95, thumb_y=0.2, index=0.15, middle=0.15)),
    4: _sequence_left(4, "tripod", _hand_left(index=0.85, middle=0.85, thumb_p=0.85, thumb_y=0.4)),
    5: _sequence_left(5, "hook", _hand_left(pinky=0.8, ring=0.8, middle=0.8, index=0.8, thumb_p=0.15)),
    6: _sequence_left(6, "precision", _hand_left(index=0.95, thumb_p=0.95, thumb_y=0.35)),
}


def get_grasp(grasp_id: int) -> GraspSequence:
    if grasp_id not in GRASPS:
        raise ValueError(f"grasp_id must be 1-6, got {grasp_id}")
    return GRASPS[grasp_id]
