import json
import os

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FacePublisher(Node):
    def __init__(self):
        super().__init__("face_publisher")
        self.bridge = CvBridge()

        # assets path: share/emotion_agent/assets/faces
        pkg_share = self._find_share_dir("emotion_agent")
        self.faces_dir = os.path.join(pkg_share, "assets", "faces")

        self.pub = self.create_publisher(Image, "/agent/face_image", 10)
        self.sub = self.create_subscription(String, "/agent/action", self._on_action, 10)

        self.cache = {}
        self.get_logger().info(f"FacePublisher using faces dir: {self.faces_dir}")

    def _find_share_dir(self, pkg: str) -> str:
        # ament_index_python is standard in ROS2
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(pkg)

    def _load_face(self, face_id: str):
        if face_id in self.cache:
            return self.cache[face_id]
        path = os.path.join(self.faces_dir, f"{face_id}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            path2 = os.path.join(self.faces_dir, "neutral.png")
            img = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

        # --- normalize to BGR8 ---
        # If grayscale (H,W), convert to BGR
        if img is not None and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If BGRA (H,W,4), drop alpha -> BGR
        elif img is not None and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # If already BGR (H,W,3), keep as is

        self.cache[face_id] = img
        return img


    def _on_action(self, msg: String):
        try:
            obj = json.loads(msg.data)
            face_id = str(obj["face_id"])
        except Exception as e:
            self.get_logger().warn(f"Bad action msg: {e}")
            return

        img = self._load_face(face_id)
        ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub.publish(ros_img)
        self.get_logger().info(f"Publish face image: {face_id}")

def main():
    rclpy.init()
    node = FacePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()