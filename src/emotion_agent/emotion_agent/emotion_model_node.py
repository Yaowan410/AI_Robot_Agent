#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# audio
import soundfile as sf
import librosa

# HF
from transformers import AutoFeatureExtractor, Wav2Vec2Model


# ----------------------------
# Model definition that matches your state_dict:
# keys:
#   wav2vec2.*
#   projector.weight/bias  (proj_out, proj_in)
#   classifier.weight/bias (num_labels, proj_out)
# ----------------------------
class Wav2Vec2ProjectorClassifier(nn.Module):
    def __init__(self, base_model_name: str, proj_out: int, num_labels: int):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(base_model_name)
        hidden = self.wav2vec2.config.hidden_size  # should match proj_in in ckpt

        self.projector = nn.Linear(hidden, proj_out)
        self.classifier = nn.Linear(proj_out, num_labels)

    def forward(self, input_values, attention_mask=None):
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B, T, H]
        pooled = hs.mean(dim=1)     # [B, H] mean pooling
        x = self.projector(pooled)  # [B, proj_out]
        logits = self.classifier(x) # [B, num_labels]
        return logits


@dataclass
class InferResult:
    emotion: str
    credibility: float


class EmotionModelNode(Node):
    """
    Publishes: /agent/emotion  (std_msgs/String JSON)
      {"emotion": "...", "credibility": 0.87}

    Params:
      - ckpt_path (string): path to your .pt (state_dict OrderedDict)
      - wav_path  (string): path to a .wav file to run inference on (looping)
      - model_name (string): HF backbone for wav2vec2
            try: facebook/wav2vec2-base   (hidden=768)
                 facebook/wav2vec2-large-960h (hidden=1024)
        This MUST match projector input dim in your ckpt.
      - period_sec (double): publish interval
      - max_duration_sec (double): truncate audio to N seconds
      - device (string): "cpu" or "cuda"
    """

    def __init__(self):
        super().__init__("emotion_model_node")

        # ---- parameters ----
        self.declare_parameter("ckpt_path", "")
        self.declare_parameter("wav_path", "")
        self.declare_parameter("model_name", "facebook/wav2vec2-base")
        self.declare_parameter("period_sec", 2.0)
        self.declare_parameter("max_duration_sec", 6.0)
        self.declare_parameter("device", "cpu")  # or "cuda"

        self.ckpt_path = self.get_parameter("ckpt_path").get_parameter_value().string_value
        self.wav_path = self.get_parameter("wav_path").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.period_sec = float(self.get_parameter("period_sec").value)
        self.max_duration_sec = float(self.get_parameter("max_duration_sec").value)

        dev_str = self.get_parameter("device").get_parameter_value().string_value.strip().lower()
        if dev_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not self.ckpt_path:
            self.get_logger().error("ckpt_path is empty. Pass: --ros-args -p ckpt_path:=/abs/path/model.pt")
        if not self.wav_path:
            self.get_logger().error("wav_path is empty. Pass: --ros-args -p wav_path:=/abs/path/test.wav")

        self.get_logger().info(f"Using device: {self.device}")

        # ---- publisher ----
        self.pub = self.create_publisher(String, "/agent/emotion", 10)

        # ---- load checkpoint (state_dict) and infer dims ----
        self.state_dict: Optional[dict] = None
        self.proj_out: Optional[int] = None
        self.proj_in: Optional[int] = None
        self.num_labels: Optional[int] = None
        self.id2label: Dict[int, str] = {}

        self.feature_extractor = None
        self.target_sr: Optional[int] = None
        self.max_len_samples: Optional[int] = None
        self.model: Optional[nn.Module] = None

        if self.ckpt_path:
            self._load_state_dict_and_build()

        # ---- timer ----
        self.timer = self.create_timer(self.period_sec, self._on_timer)
        self.get_logger().info(f"EmotionModelNode publishing /agent/emotion every {self.period_sec}s")

    def _load_state_dict_and_build(self):
        sd = torch.load(self.ckpt_path, map_location="cpu")

        if not isinstance(sd, dict):
            raise RuntimeError(f"Checkpoint is not a dict/OrderedDict: {type(sd)}")

        # required keys
        if "projector.weight" not in sd or "classifier.weight" not in sd:
            raise RuntimeError("state_dict missing projector/classifier. This node expects wav2vec2 + projector + classifier.")

        proj_out, proj_in = sd["projector.weight"].shape           # (256, 768) or (256, 1024)
        num_labels, cls_in = sd["classifier.weight"].shape         # (6, 256)

        if cls_in != proj_out:
            raise RuntimeError(f"Shape mismatch: classifier in={cls_in} but projector out={proj_out}")

        self.state_dict = sd
        self.proj_out = int(proj_out)
        self.proj_in = int(proj_in)
        self.num_labels = int(num_labels)

        # IMPORTANT: label order is NOT stored in your ckpt.
        # We assume your training used:
        # ORIG_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
        if self.num_labels == 6:
            labels = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
        else:
            labels = [f"class_{i}" for i in range(self.num_labels)]
        self.id2label = {i: labels[i] for i in range(self.num_labels)}

        # feature extractor / SR
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.target_sr = int(self.feature_extractor.sampling_rate)
        self.max_len_samples = int(self.max_duration_sec * self.target_sr)

        # build model
        self.model = Wav2Vec2ProjectorClassifier(self.model_name, self.proj_out, self.num_labels).to(self.device)

        # sanity check: backbone hidden must match projector input
        hidden = int(self.model.wav2vec2.config.hidden_size)
        if hidden != self.proj_in:
            raise RuntimeError(
                f"Backbone hidden_size={hidden} does NOT match projector input={self.proj_in}. "
                f"Your model_name is wrong. Try:\n"
                f"  -p model_name:=facebook/wav2vec2-base (hidden=768)\n"
                f"  -p model_name:=facebook/wav2vec2-large-960h (hidden=1024)"
            )

        missing, unexpected = self.model.load_state_dict(self.state_dict, strict=False)
        if missing or unexpected:
            self.get_logger().warn(f"load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}")
            self.get_logger().warn(f"missing sample: {missing[:5]}")
            self.get_logger().warn(f"unexpected sample: {unexpected[:5]}")

        self.model.eval()
        self.get_logger().info(
            f"Loaded ckpt={self.ckpt_path} | proj_out={self.proj_out} proj_in={self.proj_in} | "
            f"num_labels={self.num_labels} | id2label={self.id2label}"
        )

    def _load_wav(self, wav_path: str) -> Tuple[np.ndarray, int]:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # stereo -> mono
        return wav.astype(np.float32), int(sr)

    @torch.no_grad()
    def infer_once(self) -> Optional[InferResult]:
        if not self.model or not self.feature_extractor or not self.target_sr:
            return None
        if not self.wav_path:
            return None

        wav, sr = self._load_wav(self.wav_path)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        if self.max_len_samples and len(wav) > self.max_len_samples:
            wav = wav[: self.max_len_samples]

        inputs = self.feature_extractor(
            wav,
            sampling_rate=self.target_sr,
            padding=True,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(input_values=input_values, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [C]
        pred_id = int(torch.argmax(probs).item())
        cred = float(torch.max(probs).item())

        emotion = self.id2label.get(pred_id, f"class_{pred_id}")
        return InferResult(emotion=emotion, credibility=cred)

    def _on_timer(self):
        try:
            res = self.infer_once()
            if res is None:
                return

            msg = String()
            msg.data = json.dumps(
                {"emotion": res.emotion, "credibility": round(res.credibility, 3), "t": time.time()},
                ensure_ascii=False,
            )
            self.pub.publish(msg)
            self.get_logger().info(f"Publish emotion: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"inference failed: {e}")


def main():
    rclpy.init()
    node = None
    try:
        node = EmotionModelNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()