import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

HELP = """
Reward keyboard:
  1  -> reward +1
  0  -> reward  0
  -  -> reward -1
  q  -> quit
"""

class RewardKeyboard(Node):
    def __init__(self):
        super().__init__("reward_keyboard")
        self.pub = self.create_publisher(String, "/agent/reward", 10)
        self.get_logger().info(HELP)

    def _getch(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def run(self):
        while rclpy.ok():
            ch = self._getch()
            if ch == "q":
                self.get_logger().info("Quit.")
                return
            if ch == "1":
                self._publish(1.0)
            elif ch == "0":
                self._publish(0.0)
            elif ch == "-":
                self._publish(-1.0)

    def _publish(self, r: float):
        msg = String()
        msg.data = str(r)
        self.pub.publish(msg)
        self.get_logger().info(f"Publish reward: {r:+.1f}")

def main():
    rclpy.init()
    node = RewardKeyboard()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()