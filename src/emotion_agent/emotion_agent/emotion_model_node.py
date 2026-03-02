import json
import io
import numpy as np
import torch
from torch import nn

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from transformers import AutoFeatureExtractor, AutoModel
import soundfile as sf
import librosa


# --------------------
# Model (same as your training)
# --------------------
class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        B, T, H = hidden_states.size()
        # NOTE: your training code used all-ones mask; we keep same behavior
        mask = torch.ones(B, T, device=hidden_states.device, dtype=torch.long)

        attn_logits = self.attention(hidden_states)  # [B, T, 1]
        mask_ = mask.unsqueeze(-1)                   # [B, T, 1]
        attn_logits = attn_logits.masked_fill(mask_ == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, T, 1]

        mean = torch.sum(attn_weights * hidden_states, dim=1)  # [B, H]
        diff = hidden_states - mean.unsqueeze(1)               # [B, T, H]
        var = torch.sum(attn_weights * diff * diff, dim=1)     # [B, H]
        std = torch.sqrt(torch.clamp(var, min=1e-9))           # [B, H]

        stats = torch.cat([mean, std], dim=-1)  # [B, 2H]
        return stats


class WavLMRelabel5Classifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.wavlm.config.hidden_size
        self.pooling = AttentiveStatsPooling(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        pooled = self.pooling(hidden_states)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def load_wav_mono(wav_path: str):
    wav, sr = sf.read(wav_path)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


class EmotionModelNode(Node):
    def __init__(self):
        super().__init__("emotion_model_node")

        # -------- ROS params --------
        self.declare_parameter("ckpt_path", "")
        self.declare_parameter("wav_path", "")
        self.declare_parameter("model_name", "microsoft/wavlm-base")
        self.declare_parameter("max_duration_sec", 6.0)
        self.declare_parameter("period_sec", 2.0)

        self.ckpt_path = self.get_parameter("ckpt_path").value
        self.wav_path = self.get_parameter("wav_path").value
        self.model_name = self.get_parameter("model_name").value
        self.max_duration_sec = float(self.get_parameter("max_duration_sec").value)
        self.period = float(self.get_parameter("period_sec").value)

        if not self.ckpt_path:
            self.get_logger().error("ckpt_path is empty. Pass: --ros-args -p ckpt_path:=/abs/path/best_wavlm_relabel5.pt")
        if not self.wav_path:
            self.get_logger().error("wav_path is empty. Pass: --ros-args -p wav_path:=/abs/path/test.wav")

        # -------- Torch device --------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # -------- Load checkpoint --------
        self.ckpt = None
        self.id2label = None
        self.label2id = None
        self.num_labels = None

        if self.ckpt_path:
            self.ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self.id2label = self.ckpt["id2label"]          # dict: int->str
            self.label2id = self.ckpt.get("label2id", None)
            self.num_labels = len(self.id2label)
            base_name = self.ckpt.get("model_name", self.model_name)

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(base_name)
            self.target_sr = self.feature_extractor.sampling_rate
            self.max_len = int(self.max_duration_sec * self.target_sr)

            self.model = WavLMRelabel5Classifier(base_name, self.num_labels).to(self.device)
            self.model.load_state_dict(self.ckpt["model_state_dict"], strict=True)
            self.model.eval()

            self.get_logger().info(
                f"Loaded ckpt: {self.ckpt_path}, labels={self.id2label}, target_sr={self.target_sr}"
            )
        else:
            # still create to avoid crash; it just won't publish
            self.feature_extractor = None
            self.model = None
            self.target_sr = None
            self.max_len = None

        # -------- Publisher --------
        self.pub = self.create_publisher(String, "/agent/emotion", 10)

        # -------- Timer --------
        self.timer = self.create_timer(self.period, self.tick)
        self.get_logger().info(f"EmotionModelNode publishing /agent/emotion every {self.period}s")

        # map your "high_neg" to agent's expected "angry"
        self.label_map = {
            "high_neg": "angry",
            "excited": "excited",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
        }

    @torch.no_grad()
    def infer_once(self, wav_path: str):
        wav, sr = load_wav_mono(wav_path)

        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        if len(wav) > self.max_len:
            wav = wav[: self.max_len]

        inputs = self.feature_extractor(
            [wav],
            sampling_rate=self.target_sr,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(input_values=input_values, attention_mask=attention_mask)  # [1, C]
        probs = torch.softmax(logits, dim=-1)[0]  # [C]
        conf, pred_id = torch.max(probs, dim=-1)

        pred_label = self.id2label[int(pred_id.item())]
        credibility = float(conf.item())

        # map label
        emotion_out = self.label_map.get(pred_label, pred_label)
        return emotion_out, credibility

    def tick(self):
        if not (self.model and self.feature_extractor and self.wav_path):
            return

        try:
            emo, cred = self.infer_once(self.wav_path)
        except Exception as e:
            self.get_logger().error(f"inference failed: {e}")
            return

        payload = {"emotion": emo, "credibility": float(cred)}
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        self.get_logger().info(f"Publish /agent/emotion: {payload}")


def main():
    rclpy.init()
    node = EmotionModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()