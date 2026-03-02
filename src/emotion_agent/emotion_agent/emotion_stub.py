import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

EMOTIONS = ["neutral", "happy", "sad", "angry", "excited"]

class EmotionStub(Node):
    def __init__(self):
        super().__init__("emotion_stub")
        self.declare_parameter("publish_hz", 0.5)
        hz = float(self.get_parameter("publish_hz").value)

        self.pub = self.create_publisher(String, "/agent/emotion", 10)
        self.timer = self.create_timer(1.0 / hz, self._tick)

    def _tick(self):
        emo = random.choice(EMOTIONS)
        # 模拟置信度：neutral 偏高，其他略低一点（只是为了 demo 更像真的）
        base = 0.75 if emo == "neutral" else 0.65
        cred = max(0.05, min(0.99, random.gauss(base, 0.12)))
        msg = String()
        msg.data = f'{{"emotion":"{emo}","credibility":{cred:.3f}}}'
        self.pub.publish(msg)
        self.get_logger().info(f"Publish emotion: {msg.data}")

def main():
    rclpy.init()
    node = EmotionStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()