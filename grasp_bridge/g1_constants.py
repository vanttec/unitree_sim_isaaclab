"""G1-29DOF motor indices and arm PD gains (aligned with xr_teleoperate)."""

G1_NUM_MOTOR = 29

ARM_JOINTS = (
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
)
LEFT_ARM_JOINTS = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_JOINTS = (22, 23, 24, 25, 26, 27, 28)

_ARM_KP, _WRIST_KP = 80.0, 40.0
_ARM_KD, _WRIST_KD = 3.0, 1.5

_KP = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]
_KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]
for _i in (*LEFT_ARM_JOINTS[:4], *RIGHT_ARM_JOINTS[:4]):
    _KP[_i], _KD[_i] = _ARM_KP, _ARM_KD
for _i in (*LEFT_ARM_JOINTS[4:], *RIGHT_ARM_JOINTS[4:]):
    _KP[_i], _KD[_i] = _WRIST_KP, _WRIST_KD
KP = tuple(_KP)
KD = tuple(_KD)

INSPIRE_OPEN = (0.0,) * 12
