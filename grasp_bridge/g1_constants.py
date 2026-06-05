"""G1-29DOF joint indices and default PD gains (Unitree SDK2 convention)."""

G1_NUM_MOTOR = 29

# Right arm drives the cylinder task in the default scene.
ARM_JOINTS = (
    15, 16, 17, 18, 19, 20, 21,  # left arm
    22, 23, 24, 25, 26, 27, 28,  # right arm
)

RIGHT_ARM_JOINTS = (22, 23, 24, 25, 26, 27, 28)
LEFT_ARM_JOINTS = (15, 16, 17, 18, 19, 20, 21)

KP = (
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
)

KD = (
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
)

# Inspire DDS motor order (rt/inspire/cmd): right hand [0-5], left hand [6-11].
# Indices: pinky, ring, middle, index, thumb_pitch, thumb_yaw
INSPIRE_OPEN = (0.0,) * 12
