import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros_gz_interfaces.srv import SetEntityPose


FACE_MODEL = {
    "happy": "simple_face_happy",
    "sad": "simple_face_sad",
    "angry": "simple_face_angry",
    "neutral": "simple_face_neutral",
    "excited": "simple_face_excited",
}

FAR = (99.0, 99.0, 1.0)

# fixed base pose (standing)
BASE_X, BASE_Y = 0.0, 0.0
BODY_Z = 0.28
HEAD_Z = 0.72
ARM_Z  = 0.52

LEFT_ARM_X  = -0.28
RIGHT_ARM_X =  0.28

# face offset: slightly in front of head (y axis thickness is small; we just place it in +x direction)
FACE_X_OFFSET = 0.18
FACE_Z_OFFSET = 0.72  # roughly head center height


def yaw_to_quat(yaw: float):
    # roll=pitch=0
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))  # x,y,z,w


class SimpleBotController(Node):
    def __init__(self):
        super().__init__("gz_simple_bot_controller")
        self.declare_parameter("world", "default")
        self.world = self.get_parameter("world").value

        self.current_face = "neutral"
        self.current_body = "idle"
        self.last_action_ts = time.time()

        srv = f"/world/{self.world}/set_pose"
        self.cli = self.create_client(SetEntityPose, srv)
        self.get_logger().info(f"Waiting for service {srv} ...")
        self.cli.wait_for_service()
        self.get_logger().info("Connected to set_pose service.")

        self.sub = self.create_subscription(String, "/agent/action", self.on_action, 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20Hz animation

        # Initial placement
        self.place_static_parts()
        self.apply_face_models()

    def call_set_pose(self, name: str, x: float, y: float, z: float, yaw: float):
        req = SetEntityPose.Request()
        req.name = name
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw
        self.cli.call_async(req)

    def place_static_parts(self):
        # Keep body fixed
        self.call_set_pose("simple_bot_body", BASE_X, BASE_Y, BODY_Z, 0.0)
        # Head is animated in tick()
        self.call_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z, 0.0)
        # Arms are animated in tick()
        self.call_set_pose("simple_bot_arm_left",  LEFT_ARM_X,  BASE_Y, ARM_Z, 0.0)
        self.call_set_pose("simple_bot_arm_right", RIGHT_ARM_X, BASE_Y, ARM_Z, 0.0)

    def on_action(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        face = data.get("face", "neutral")
        body = data.get("body", "idle")

        if face in FACE_MODEL:
            self.current_face = face
        self.current_body = body
        self.last_action_ts = time.time()

        # When face changes, update immediately
        self.apply_face_models()

    def apply_face_models(self):
        active = FACE_MODEL.get(self.current_face, FACE_MODEL["neutral"])

        # place active face in front of head
        fx = BASE_X + FACE_X_OFFSET
        fy = BASE_Y
        fz = FACE_Z_OFFSET
        self.call_set_pose(active, fx, fy, fz, 0.0)

        # hide others
        for _, model in FACE_MODEL.items():
            if model == active:
                continue
            self.call_set_pose(model, FAR[0], FAR[1], FAR[2], 0.0)

    def tick(self):
        # simple “upper body” animation based on current_body
        t = time.time()
        dt = t - self.last_action_ts

        # decay animation after ~1.2s (like your controller timing)
        strength = max(0.0, 1.2 - dt) / 1.2

        head_yaw = 0.0
        arm_left_yaw = 0.0
        arm_right_yaw = 0.0

        if self.current_body == "nod":
            # nod ≈ small forward/back motion (yaw is limited; we fake with tiny yaw + vertical bob)
            bob = 0.03 * math.sin(10.0 * t) * strength
            self.call_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z + bob, 0.0)
        elif self.current_body == "shake":
            head_yaw = 0.4 * math.sin(10.0 * t) * strength
            self.call_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z, head_yaw)
        elif self.current_body == "wave":
            # wave right arm
            arm_right_yaw = 1.2 * math.sin(10.0 * t) * strength
            self.call_set_pose("simple_bot_arm_right", RIGHT_ARM_X, BASE_Y, ARM_Z, arm_right_yaw)
            self.call_set_pose("simple_bot_arm_left",  LEFT_ARM_X,  BASE_Y, ARM_Z, 0.0)
            self.call_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z, 0.0)
        else:
            # idle / approach/retreat/turn_* -> since we don't move, we just do a subtle “attention” motion
            bob = 0.01 * math.sin(6.0 * t) * strength
            self.call_set_pose("simple_bot_head", BASE_X, BASE_Y, HEAD_Z + bob, 0.0)

        # keep face pinned in front of head (always)
        self.apply_face_models()


def main():
    rclpy.init()
    node = SimpleBotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()