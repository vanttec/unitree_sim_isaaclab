"""Execute a grasp sequence by interpolating keyframes over DDS."""

from __future__ import annotations

import time

import numpy as np

from .dds_client import UnitreeDdsClient
from .g1_constants import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from .grasp_library import get_grasp


class GraspExecutor:
    def __init__(self, client: UnitreeDdsClient, control_hz: float = 50.0):
        self._client = client
        self._dt = 1.0 / control_hz

    def run(self, grasp_id: int) -> None:
        sequence = get_grasp(grasp_id)
        side = sequence.active_side
        print(f"[grasp] Running #{grasp_id} ({sequence.label}) — side={side}")

        body_q = self._client.current_joint_positions()
        left_q = np.array([body_q[i] for i in LEFT_ARM_JOINTS], dtype=np.float64)
        right_q = np.array([body_q[i] for i in RIGHT_ARM_JOINTS], dtype=np.float64)
        hand_q = np.zeros(12, dtype=np.float64)

        for kf in sequence.keyframes:
            print(f"  -> phase: {kf.name} ({kf.duration_s:.1f}s)")
            left_goal = np.array(kf.left_arm, dtype=np.float64)
            right_goal = np.array(kf.right_arm, dtype=np.float64)

            if side == "left":
                left_target, right_target = left_goal, right_q.copy()
            else:
                left_target, right_target = left_q.copy(), right_goal

            hand_goal = np.array(kf.hand_q, dtype=np.float64)
            steps = max(1, int(kf.duration_s / self._dt))

            for step in range(steps):
                alpha = (step + 1) / steps
                left_cmd = (1.0 - alpha) * left_q + alpha * left_target
                right_cmd = (1.0 - alpha) * right_q + alpha * right_target
                hand_cmd = (1.0 - alpha) * hand_q + alpha * hand_goal

                body_q = self._client.current_joint_positions()
                for i, j in enumerate(LEFT_ARM_JOINTS):
                    body_q[j] = left_cmd[i]
                for i, j in enumerate(RIGHT_ARM_JOINTS):
                    body_q[j] = right_cmd[i]

                self._client.publish_lowcmd(body_q)
                self._client.publish_inspire(hand_cmd)
                time.sleep(self._dt)

            left_q, right_q = left_target, right_target
            hand_q = hand_goal

        print(f"[grasp] Done: {sequence.label}")
