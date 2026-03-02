import json
import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


WORLD = "empty"

# base poses
BASE_X, BASE_Y = 0.0, 0.0
BODY_Z = 0.28
HEAD_Z = 0.72
ARM_Z = 0.52
LEFT_ARM_X  = -0.28
RIGHT_ARM_X =  0.28

def gz_set_pose(name: str, x: float, y: float, z: float, roll=0.0, pitch=0.0, yaw=0.0):
    # gz.msgs.Pose uses position + orientation quaternion, but CLI accepts Euler in some builds inconsistently.
    # To be robust, we only set position most of the time; for rotation we approximate via quaternion fields.
    # Many setups accept: orientation { x: .. y: .. z: .. w: .. }
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    req = f'''name: "{name}"
position {{ x: {x} y: {y} z: {z} }}
orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}
'''
    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD}/set_pose/blocking",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "10000",
        "--req", req
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class AvatarMotion(Node):
    def __init__(self):
        super().__init__("gz_avatar_motion")
        self.current_body = "idle"
        self.last_action_ts = time.time()

        self.sub = self.create_subscription(String, "/agent/action", self.on_action, 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20Hz

        # place static parts (in case)
        gz_set_pose("simple_bot_body", BASE_X, BASE_Y, BODY_Z)
        gz_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z)
        gz_set_pose("simple_bot_arm_left",  LEFT_ARM_X,  BASE_Y, ARM_Z)
        gz_set_pose("simple_bot_arm_right", RIGHT_ARM_X, BASE_Y, ARM_Z)

        self.get_logger().info("gz_avatar_motion running (subscribing /agent/action).")

    def on_action(self, msg: String):
        self.get_logger().info(f"RAW msg.data = {repr(msg.data)}")
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"json.loads failed: {e}")
            return

        body = data.get("body_id", "idle")
        self.get_logger().info(f"PARSED body_id = {body}")
        self.current_body = body
        self.last_action_ts = time.time()

    def tick(self):
        t = time.time()
        dt = t - self.last_action_ts
        strength = max(0.0, 1.2 - dt) / 1.2  # fade out

        # reset default
        head_roll = 0.0
        head_pitch = 0.0
        head_yaw = 0.0
        arm_r_yaw = 0.0

        if self.current_body == "nod":
            head_pitch = 0.5 * math.sin(10.0 * t) * strength
        elif self.current_body == "shake":
            head_yaw = 0.6 * math.sin(10.0 * t) * strength
        elif self.current_body == "wave":
            arm_r_yaw = 1.2 * math.sin(10.0 * t) * strength

        # apply poses
        gz_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z, roll=head_roll, pitch=head_pitch, yaw=head_yaw)
        gz_set_pose("simple_bot_arm_right", RIGHT_ARM_X, BASE_Y, ARM_Z, yaw=arm_r_yaw)
        gz_set_pose("simple_bot_arm_left",  LEFT_ARM_X,  BASE_Y, ARM_Z)

def main():
    rclpy.init()
    node = AvatarMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()