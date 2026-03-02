import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

EMOTIONS = ["neutral", "happy", "sad", "angry", "excited"]

# face_id 和 body_id 的可选集合
FACE_IDS = ["neutral", "happy", "sad", "angry", "excited"]
BODY_IDS_ALL = ["idle", "nod", "shake", "approach", "retreat", "turn_left", "turn_right"]

# 低置信度下的保守动作集合（gating）
BODY_IDS_SAFE = ["idle", "nod"]

@dataclass(frozen=True)
class Action:
    face_id: str
    body_id: str

def one_hot(emotion: str) -> List[float]:
    return [1.0 if e == emotion else 0.0 for e in EMOTIONS]

class LinearBandit:
    """
    Very simple contextual bandit:
      score(a|x) = w_a · x
    Update with incremental gradient step:
      w_a <- w_a + alpha * (reward - w_a·x) * x
    """
    def __init__(self, dim: int, alpha: float = 0.25):
        self.dim = dim
        self.alpha = alpha
        self.w: Dict[Action, List[float]] = {}

    def _ensure(self, a: Action):
        if a not in self.w:
            self.w[a] = [0.0] * self.dim

    def score(self, a: Action, x: List[float]) -> float:
        self._ensure(a)
        wa = self.w[a]
        return sum(wa[i] * x[i] for i in range(self.dim))

    def update(self, a: Action, x: List[float], r: float):
        self._ensure(a)
        wa = self.w[a]
        pred = sum(wa[i] * x[i] for i in range(self.dim))
        err = r - pred
        for i in range(self.dim):
            wa[i] += self.alpha * err * x[i]

class BanditAgent(Node):
    def __init__(self):
        super().__init__("bandit_agent")
        self.declare_parameter("credibility_threshold", 0.55)
        self.declare_parameter("epsilon", 0.25)
        self.declare_parameter("alpha", 0.25)
        self.declare_parameter("log_path", "/tmp/emotion_agent_log.jsonl")

        self.cred_th = float(self.get_parameter("credibility_threshold").value)
        self.epsilon = float(self.get_parameter("epsilon").value)
        alpha = float(self.get_parameter("alpha").value)
        self.log_path = str(self.get_parameter("log_path").value)

        # context dim = one_hot(emotion)=5 + credibility=1 + bias=1 => 7
        self.dim = len(EMOTIONS) + 1 + 1
        self.bandit = LinearBandit(dim=self.dim, alpha=alpha)

        self.emotion_sub = self.create_subscription(String, "/agent/emotion", self._on_emotion, 10)
        self.reward_sub = self.create_subscription(String, "/agent/reward", self._on_reward, 10)

        self.action_pub = self.create_publisher(String, "/agent/action", 10)

        self.last_context = None
        self.last_action = None
        self.last_step_ts = None
        self.step_idx = 0

        self.get_logger().info("BanditAgent started.")

    def _parse_emotion_msg(self, s: str) -> Tuple[str, float]:
        obj = json.loads(s)
        emo = str(obj["emotion"])
        cred = float(obj["credibility"])
        if emo not in EMOTIONS:
            emo = "neutral"
        cred = max(0.0, min(1.0, cred))
        return emo, cred

    def _make_context(self, emo: str, cred: float) -> List[float]:
        x = one_hot(emo) + [cred] + [1.0]  # bias
        assert len(x) == self.dim
        return x

    def _available_actions(self, cred: float) -> List[Action]:
        body_ids = BODY_IDS_SAFE if cred < self.cred_th else BODY_IDS_ALL
        return [Action(face_id=f, body_id=b) for f in FACE_IDS for b in body_ids]

    def _epsilon_greedy(self, actions: List[Action], x: List[float]) -> Action:
        import random
        if random.random() < self.epsilon:
            return random.choice(actions)
        # exploit: choose best score
        best_a = actions[0]
        best_s = -1e9
        for a in actions:
            s = self.bandit.score(a, x)
            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    def _publish_action(self, a: Action, emo: str, cred: float):
        msg = String()
        msg.data = json.dumps({
            "face_id": a.face_id,
            "body_id": a.body_id,
            "emotion": emo,
            "credibility": cred,
            "step": self.step_idx,
            "t": time.time(),
        })
        self.action_pub.publish(msg)
        self.get_logger().info(f"STEP {self.step_idx}: choose action face={a.face_id} body={a.body_id} (emo={emo}, cred={cred:.2f})")

    def _log(self, record: dict):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to write log: {e}")

    def _on_emotion(self, msg: String):
        emo, cred = self._parse_emotion_msg(msg.data)
        x = self._make_context(emo, cred)
        actions = self._available_actions(cred)
        a = self._epsilon_greedy(actions, x)

        self.last_context = x
        self.last_action = a
        self.last_step_ts = time.time()

        self._publish_action(a, emo, cred)

        self._log({
            "type": "step",
            "step": self.step_idx,
            "emotion": emo,
            "credibility": cred,
            "context": x,
            "action": {"face_id": a.face_id, "body_id": a.body_id},
            "t": self.last_step_ts,
        })

        self.step_idx += 1

    def _on_reward(self, msg: String):
        if self.last_action is None or self.last_context is None:
            self.get_logger().warn("Got reward but no last_action/context yet.")
            return
        try:
            r = float(msg.data)
            r = max(-1.0, min(1.0, r))
        except Exception:
            self.get_logger().warn(f"Bad reward: {msg.data}")
            return

        # update
        a = self.last_action
        x = self.last_context
        self.bandit.update(a, x, r)

        self.get_logger().info(f"Update bandit with reward={r:+.1f} for action face={a.face_id} body={a.body_id}")

        self._log({
            "type": "reward",
            "reward": r,
            "action": {"face_id": a.face_id, "body_id": a.body_id},
            "context": x,
            "t": time.time(),
        })

def main():
    rclpy.init()
    node = BanditAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()