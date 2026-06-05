#!/usr/bin/env python3
"""ROS2 front-end for the Unitree grasp bridge.

Usage:
    # Terminal 1: sim
    python sim_main.py --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \\
        --enable_inspire_dds --robot_type g129 --enable_cameras

    # Terminal 2: bridge
    source /opt/ros/humble/setup.bash
    conda activate env_isaaclab
    python -m grasp_bridge.ros2_node

    # Terminal 3: trigger
    ros2 topic pub --once /grasp_command std_msgs/msg/Int32 "{data: 2}"
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String

from .dds_client import UnitreeDdsClient
from .executor import GraspExecutor
from .grasp_library import GRASPS


class GraspBridgeNode(Node):
    def __init__(self):
        super().__init__("unitree_grasp_bridge")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("dds_channel", 1)

        hz = self.get_parameter("control_hz").value
        channel = self.get_parameter("dds_channel").value

        self._client = UnitreeDdsClient(domain_channel=channel)
        self._executor = GraspExecutor(self._client, control_hz=hz)
        self._busy = threading.Lock()

        self._status_pub = self.create_publisher(String, "/grasp_status", 10)
        self.create_subscription(Int32, "/grasp_command", self._on_grasp_command, 10)

        labels = ", ".join(f"{k}={v.label}" for k, v in sorted(GRASPS.items()))
        self.get_logger().info(f"Ready. Publish Int32 1-6 on /grasp_command. Grasps: {labels}")

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)
        self.get_logger().info(text)

    def _on_grasp_command(self, msg: Int32) -> None:
        grasp_id = int(msg.data)
        if grasp_id not in GRASPS:
            self.get_logger().warn(f"Invalid grasp_id {grasp_id}; use 1-6")
            return

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn("Grasp already running, ignoring command")
            return

        def _run() -> None:
            try:
                self._publish_status(f"running:{grasp_id}")
                self._executor.run(grasp_id)
                self._publish_status(f"done:{grasp_id}")
            except Exception as exc:
                self.get_logger().error(f"Grasp failed: {exc}")
                self._publish_status(f"error:{grasp_id}:{exc}")
            finally:
                self._busy.release()

        threading.Thread(target=_run, daemon=True).start()


def main() -> None:
    rclpy.init()
    node = GraspBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
