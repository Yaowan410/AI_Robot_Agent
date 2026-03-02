import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class ExpressionController(Node):
    def __init__(self):
        super().__init__("expression_controller")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("action_duration_sec", 1.2)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.duration = float(self.get_parameter("action_duration_sec").value)

        self.sub = self.create_subscription(String, "/agent/action", self._on_action, 10)
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.timer = None
        self.stop_at = None
        self.get_logger().info(f"ExpressionController publishing Twist to {self.cmd_vel_topic}")

    def _publish_twist(self, lin_x: float, ang_z: float):
        t = Twist()
        t.linear.x = float(lin_x)
        t.angular.z = float(ang_z)
        self.pub.publish(t)

    def _stop(self):
        self._publish_twist(0.0, 0.0)

    def _on_action(self, msg: String):
        try:
            obj = json.loads(msg.data)
            body_id = str(obj["body_id"])
        except Exception as e:
            self.get_logger().warn(f"Bad action msg: {e}")
            return

        # Cancel previous timer if any
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        # Map body action to Twist
        lin, ang = 0.0, 0.0
        if body_id == "idle":
            lin, ang = 0.0, 0.0
        elif body_id == "nod":
            # nod: short forward/back micro-move
            lin, ang = 0.08, 0.0
        elif body_id == "shake":
            # shake: rotate left-right a bit
            lin, ang = 0.0, 0.8
        elif body_id == "approach":
            lin, ang = 0.20, 0.0
        elif body_id == "retreat":
            lin, ang = -0.18, 0.0
        elif body_id == "turn_left":
            lin, ang = 0.0, 1.0
        elif body_id == "turn_right":
            lin, ang = 0.0, -1.0
        else:
            lin, ang = 0.0, 0.0

        self.get_logger().info(f"Execute body action: {body_id} -> (lin={lin:.2f}, ang={ang:.2f}) for {self.duration:.1f}s")
        self._publish_twist(lin, ang)

        self.stop_at = time.time() + self.duration
        self.timer = self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.stop_at is None:
            return
        if time.time() >= self.stop_at:
            self._stop()
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            self.stop_at = None

def main():
    rclpy.init()
    node = ExpressionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()