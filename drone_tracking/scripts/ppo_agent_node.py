#!/usr/bin/env python3
"""
ppo_agent_node.py — PPO v4.5 Agent (normalized observations)
==============================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics

Matches PPO v4.5 training with NORMALIZED observations:
  - Network input: [ex, ey, alpha_real / 0.06]
  - Network output: [alpha_norm, lambda_norm] clipped to [-1, 1]
  - Rescaled to [0, 1] then decoded to alpha* and lambda

Subscribes: /drone_tracking/filtered_target  (geometry_msgs/Point from Kalman)
Publishes:  /drone_tracking/ibvs_setpoints   (geometry_msgs/Quaternion: x=x*, y=y*, z=alpha*, w=lambda)
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
from geometry_msgs.msg import Point, Quaternion


class PPOPolicyNetwork(nn.Module):
    """
    SB3 MlpPolicy actor reconstruction.
    Architecture: 3 → 256 → 256 → 2, Tanh activations.
    (Matches v4.5 training with net_arch=[256, 256])
    """
    def __init__(self, obs_dim=3, act_dim=2, net_arch=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in net_arch:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(prev_dim, act_dim)

    def forward(self, obs):
        features = self.policy_net(obs)
        return self.action_net(features)


class PPOAgentNode:
    ALPHA_MIN_REAL = 0.005
    ALPHA_MAX_REAL = 0.040
    ALPHA_OBS_MAX  = 0.060

    def __init__(self):
        rospy.init_node('ppo_agent_node', anonymous=True)

        weights_path = rospy.get_param(
            '~weights_path',
            '/home/rawad/drone_detection/models/ppo_policy_weights_v4.pth')

        rospy.loginfo(f"PPO Agent v4.5 — Loading weights from {weights_path}")

        self.device = torch.device('cpu')
        self.policy = PPOPolicyNetwork(obs_dim=3, act_dim=2, net_arch=[256, 256])

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Map SB3 keys → our network keys
        mapped_dict = {}
        for key, value in state_dict.items():
            if 'mlp_extractor.policy_net' in key:
                new_key = key.replace('mlp_extractor.policy_net', 'policy_net')
                mapped_dict[new_key] = value
            elif key.startswith('action_net'):
                mapped_dict[key] = value

        self.policy.load_state_dict(mapped_dict, strict=False)
        self.policy.eval()
        rospy.loginfo("✓ PPO v4.5 policy loaded (normalized obs, [256,256] net)")

        # Publish Quaternion on /drone_tracking/ibvs_setpoints (matches IBVS subscriber)
        self.pub = rospy.Publisher(
            '/drone_tracking/ibvs_setpoints',
            Quaternion, queue_size=1)

        # Subscribe to Kalman filter output (geometry_msgs/Point)
        self.sub = rospy.Subscriber(
            '/drone_tracking/filtered_target',
            Point, self.callback, queue_size=1)

        rospy.loginfo("PPO Agent v4.5 running — waiting for filtered_target...")

    def callback(self, msg):
        # msg is geometry_msgs/Point — access fields directly
        # Kalman publishes absolute pixel cx/cy in x/y, and bbox AREA in z
        # We need to convert cx/cy → normalized errors, and area → alpha
        img_w, img_h = 640.0, 480.0
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        area_norm = img_w * img_h  # 307200

        # If z is NaN or non-positive-with-NaN, skip
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            return

        # Normalize pixel error to [-1, 1]
        ex = (msg.x - img_cx) / img_cx
        ey = (msg.y - img_cy) / img_cy

        # Kalman z convention: z > 0 = real detection (bbox area in pixels)
        #                     z < 0 = prediction (negative bbox area)
        # Use absolute value to get alpha regardless of prediction flag
        alpha = abs(msg.z) / area_norm

        # NORMALIZE alpha to match training observation space
        alpha_norm_obs = alpha / self.ALPHA_OBS_MAX  # → [0, 1]

        obs = np.array([
            np.clip(ex,              -1.0, 1.0),
            np.clip(ey,              -1.0, 1.0),
            np.clip(alpha_norm_obs,   0.0, 1.0),
        ], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            action_raw = self.policy(obs_tensor).squeeze(0).numpy()

        # Clip to [-1, 1] and rescale to [0, 1]
        action_clipped = np.clip(action_raw, -1.0, 1.0)
        action_01 = (action_clipped + 1.0) / 2.0

        # Decode
        alpha_norm_out = action_01[0]
        lam            = float(action_01[1])
        alpha_star = self.ALPHA_MIN_REAL + alpha_norm_out * (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)

        # x* and y* fixed at 0 (IBVS handles centering)
        x_star = 0.0
        y_star = 0.0

        # Publish as Quaternion: x=x*, y=y*, z=alpha*, w=lambda
        out = Quaternion()
        out.x = float(x_star)
        out.y = float(y_star)
        out.z = float(alpha_star)
        out.w = float(lam)
        self.pub.publish(out)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = PPOAgentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass