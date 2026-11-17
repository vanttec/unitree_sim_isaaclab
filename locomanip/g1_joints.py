"""G1 29-DoF body joints in MuJoCo/MJCF actuator order: the order SONIC's Bones-SEED CSVs use
(gear_sonic/data_process/convert_soma_csv_to_motion_lib.py, BONES_JOINT_NAMES without "_dof")."""

BODY_JOINTS_MJ = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
assert len(BODY_JOINTS_MJ) == 29

# rt/lowcmd motor index i == BODY_JOINTS_MJ[i] (Unitree SDK order is the MJCF order)
ARM_LOWCMD_SLICE = slice(15, 29)

RECORD_TOPIC = "rt/locomanip/record_cmd"
