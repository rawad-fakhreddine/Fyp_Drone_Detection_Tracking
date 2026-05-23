#!/usr/bin/env python3
"""
ppo_agent_node.py — PPO v5.2 Agent (lambda rescaling fix)
=========================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v5.2 changes from v5.1 (1 surgical fix — no weight changes):
  * Lambda output rescaled:  lam = action_01[1]
                          →  lam = 0.3 + 0.7 * action_01[1]
    Result: lam ∈ [0.3, 1.0] instead of [0.0, 1.0]

Root cause (confirmed from flight_log_latest.csv):
  PPO was trained expecting lambda ∈ [0.3, 1.0] but deployed with
  action_01[1] ∈ [0.0, 1.0]. The bottom 30% of the PPO output range
  (0.0–0.3) was being clamped away by IBVS, so PPO learned to freely
  use that region. Observed HOLD mean = 0.236, which IBVS clamped to 0.3
  → lam_gain = 0.51 → chaser moved at 0.71 m/s vs target 0.84 m/s.
  Rescaling maps PPO's full range to [0.3, 1.0] with no retraining.

Same weights: ppo_policy_weights_v5.pth (unchanged)
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
from geometry_msgs.msg import Point, Quaternion


class PPOPolicyNetwork(nn.Module):
    """SB3 MlpPolicy actor: 6 → 256 → 256 → 2, Tanh."""
    def __init__(self, obs_dim=6, act_dim=2, net_arch=[256, 256]):
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
    # Must match training env exactly
    ALPHA_MIN_REAL = 0.003
    ALPHA_MAX_REAL = 0.020
    ALPHA_OBS_MAX  = 0.060

    # Rate amplification — must match training
    D_EX_SCALE    = 10.0
    D_EY_SCALE    = 10.0
    D_ALPHA_SCALE = 100.0

    def __init__(self):
        rospy.init_node('ppo_agent_node', anonymous=True)

        weights_path = rospy.get_param(
            '~weights_path',
            '/home/rawad/drone_detection/models/ppo_policy_weights_v5.pth')

        rospy.loginfo("PPO Agent v5.2 — Loading weights from %s" % weights_path)

        self.device = torch.device('cpu')
        self.policy = PPOPolicyNetwork(obs_dim=6, act_dim=2, net_arch=[256, 256])

        state_dict = torch.load(weights_path, map_location=self.device,
                                weights_only=True)

        mapped_dict = {}
        for key, value in state_dict.items():
            if 'mlp_extractor.policy_net' in key:
                new_key = key.replace('mlp_extractor.policy_net', 'policy_net')
                mapped_dict[new_key] = value
            elif key.startswith('action_net'):
                mapped_dict[key] = value

        self.policy.load_state_dict(mapped_dict, strict=False)
        self.policy.eval()
        rospy.loginfo("PPO v5.2 policy loaded (6D obs, [256,256], "
                      "alpha [%.3f, %.3f], lambda [0.3, 1.0])" %
                      (self.ALPHA_MIN_REAL, self.ALPHA_MAX_REAL))

        # Previous state for rate computation
        self.prev_ex = None
        self.prev_ey = None
        self.prev_alpha_norm = None
        self.prev_time = None

        self.pub = rospy.Publisher(
            '/drone_tracking/ibvs_setpoints',
            Quaternion, queue_size=1)

        self.sub = rospy.Subscriber(
            '/drone_tracking/filtered_target',
            Point, self.callback, queue_size=1)

        rospy.loginfo("PPO Agent v5.2 running — waiting for filtered_target...")

    def callback(self, msg):
        if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
            return

        img_w, img_h = 640.0, 480.0
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        area_norm = img_w * img_h

        ex = (msg.x - img_cx) / img_cx
        ey = (msg.y - img_cy) / img_cy
        alpha = abs(msg.z) / area_norm
        alpha_norm = alpha / self.ALPHA_OBS_MAX

        now = rospy.Time.now().to_sec()

        # Rate computation via finite differences
        if self.prev_ex is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if 0.01 < dt < 0.5:
                d_ex = (ex - self.prev_ex) / (dt * 20.0)
                d_ey = (ey - self.prev_ey) / (dt * 20.0)
                d_alpha_norm = (alpha_norm - self.prev_alpha_norm) / (dt * 20.0)
            else:
                d_ex = d_ey = d_alpha_norm = 0.0
        else:
            d_ex = d_ey = d_alpha_norm = 0.0

        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_alpha_norm = alpha_norm
        self.prev_time = now

        obs = np.array([
            np.clip(ex, -1.0, 1.0),
            np.clip(ey, -1.0, 1.0),
            np.clip(alpha_norm, 0.0, 1.0),
            np.clip(d_ex * self.D_EX_SCALE, -1.0, 1.0),
            np.clip(d_ey * self.D_EY_SCALE, -1.0, 1.0),
            np.clip(d_alpha_norm * self.D_ALPHA_SCALE, -1.0, 1.0),
        ], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            action_raw = self.policy(obs_tensor).squeeze(0).numpy()

        action_clipped = np.clip(action_raw, -1.0, 1.0)
        action_01 = (action_clipped + 1.0) / 2.0

        alpha_star = self.ALPHA_MIN_REAL + action_01[0] * \
                     (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)

        # v5.2 fix: rescale lambda to [0.3, 1.0] instead of [0.0, 1.0]
        # PPO's full output range now maps meaningfully onto IBVS's clamp range
        lam = 0.3 + 0.7 * float(action_01[1])

        out = Quaternion()
        out.x = 0.0
        out.y = 0.0
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