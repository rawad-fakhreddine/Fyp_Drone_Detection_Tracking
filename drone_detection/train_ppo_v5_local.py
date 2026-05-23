#!/usr/bin/env python3
"""
Train PPO v5.1 — Drone Tracking (LOCAL / WSL2)
================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

CHANGES FROM v5.0:
  1. Physics scales derived from camera geometry (focal=500px, 640x480, d≈3m):
     - VY_TO_EX_SCALE: 2.5 → 0.52  (was 4.8x too high — caused oscillation)
     - VZ_TO_EY_SCALE: 2.5 → 0.52
     - VX_TO_ALPHA_SCALE: 0.006 → 0.00454  (was 1.3x too high)
  2. Target motion recalibrated to match real-world drone speeds:
     - Hard: 0.4-0.6 m/s lateral (was too fast in pixel units)
     - Extreme: 0.6-0.8 m/s lateral (chaser barely keeps up — realistic)
     - Depth motion ~100x smaller than lateral (correct geometry)
  3. Longer episodes (500 steps max) — target should survive and chaser should
     track for extended periods, matching 25-second real flights.
  4. Dropout recovery: agent should learn to re-acquire after brief loss.
  5. 4-phase curriculum 3.4M total steps (800k/800k/1000k/800k).

At extreme difficulty, the chaser BARELY keeps up (ratio 1.05:1). This means:
  - The target WILL escape sometimes — the PPO must learn when to push harder
  - The chaser must learn to increase alpha* (follow closer) for fast targets
  - Lambda should increase when maneuvering (trust controller aggressiveness)

Run:  cd ~/catkin_ws/src/drone_detection && python3 train_ppo_v5_local.py
Out:  ~/drone_detection/models/ppo_policy_weights_v5.pth
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import gc
import math
import torch
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# =================================================================
#  ENVIRONMENT (v5.1 — correct physics)
# =================================================================
class DroneTrackingEnv(gym.Env):
    """
    6D observation, camera-geometry-calibrated physics, smooth-arc target.

    Observation: [ex, ey, alpha_norm, d_ex, d_ey, d_alpha_norm]
    Action: [alpha_norm, lambda_norm] in [-1, 1]
    """
    metadata = {"render_modes": []}

    # ── Alpha parameters (IBVS v6.15) ────────────────────────────────
    ALPHA_MIN_REAL  = 0.003
    ALPHA_MAX_REAL  = 0.020
    TARGET_ALPHA    = 0.0067
    ALPHA_OBS_MAX   = 0.060
    ALPHA_TOLERANCE = 0.002

    # ── IBVS v6.15 gains ─────────────────────────────────────────────
    K_FAR   = 14.0
    K_NEAR  = 6.0
    Kp_y    = 1.4
    Kp_z    = 1.8
    MAX_VX  = 0.70
    MAX_VX_RETREAT = 0.50
    MAX_VY  = 0.85
    MAX_VZ  = 0.80
    DT      = 1.0 / 20.0

    # ── Physics scales (derived from camera geometry) ─────────────────
    # Camera: 640x480, focal≈500px, hold distance≈3m
    # vy (m/s) → ex change: vy * focal / d * dt / (img_w/2)
    #   = vy * 500/3 * 0.05 / 320 = vy * 0.026
    # In the env: ex -= vy * DT * VY_TO_EX_SCALE
    #   = vy * 0.05 * VY_TO_EX → need VY_TO_EX = 0.026/0.05 = 0.52
    VY_TO_EX_SCALE    = 0.52
    VZ_TO_EY_SCALE    = 0.52
    # vx → alpha: alpha * ((d/(d-vx*dt))^2 - 1) / (vx*dt)
    #   At d=3, vx=0.7: alpha*(3/2.965)^2 -1) = alpha*0.0237
    #   Per step: 0.0067*0.0237 = 0.000159
    #   = vx * DT * VX_TO_ALPHA → VX_TO_ALPHA = 0.000159/(0.7*0.05) = 0.00454
    VX_TO_ALPHA_SCALE = 0.00454

    def __init__(self, difficulty='easy'):
        super().__init__()
        assert difficulty in ('easy', 'medium', 'hard', 'extreme')
        self.difficulty = difficulty

        # 6D observation
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
        )

        # ── Target motion (calibrated to real drone speeds) ───────────
        # speed: pixel-equivalent ex/ey displacement per step
        # 0.8 m/s at d=3m → 0.021/step; 0.4 m/s → 0.010/step
        # vz: alpha change per step (depth motion)
        # 0.4 m/s at d=3m → 0.00009/step (very small — correct!)
        self.motion_params = {
            'easy':    {'speed_min': 0.000, 'speed_max': 0.000,
                        'omega_max': 0.0,
                        'vz_min': 0.0, 'vz_max': 0.0,
                        'resample_min': 100, 'resample_max': 200},
            'medium':  {'speed_min': 0.003, 'speed_max': 0.010,
                        'omega_max': 0.015,
                        'vz_min': -0.00005, 'vz_max': 0.00005,
                        'resample_min': 100, 'resample_max': 300},
            'hard':    {'speed_min': 0.008, 'speed_max': 0.016,
                        'omega_max': 0.04,
                        'vz_min': -0.00009, 'vz_max': 0.00009,
                        'resample_min': 80, 'resample_max': 240},
            'extreme': {'speed_min': 0.013, 'speed_max': 0.021,
                        'omega_max': 0.06,
                        'vz_min': -0.00012, 'vz_max': 0.00012,
                        'resample_min': 60, 'resample_max': 200},
        }

        self.max_steps     = 500     # 25 seconds at 20Hz
        self.dropout_limit = 40      # 2 seconds of continuous loss

        # State
        self._state_real = None
        self._prev_state_real = None
        self._step_count = 0
        self._dropout_count = 0

        # Target motion
        self._target_heading = 0.0
        self._target_omega = 0.0
        self._target_speed = 0.0
        self._target_vz = 0.0
        self._next_omega_step = 0
        self._next_speed_step = 0
        self._next_vz_step = 0

    def _get_obs(self):
        ex, ey, alpha_real = self._state_real
        alpha_norm = alpha_real / self.ALPHA_OBS_MAX

        if self._prev_state_real is not None:
            prev_ex, prev_ey, prev_alpha = self._prev_state_real
            d_ex = ex - prev_ex
            d_ey = ey - prev_ey
            d_alpha_norm = (alpha_real - prev_alpha) / self.ALPHA_OBS_MAX
        else:
            d_ex = d_ey = d_alpha_norm = 0.0

        # Amplify rates for network visibility
        # With correct physics, typical d_ex ≈ 0.01-0.02/step at hard mode
        # Amplify by 10x → 0.1-0.2 range, comfortably in [-1,1]
        d_ex = np.clip(d_ex * 10.0, -1.0, 1.0)
        d_ey = np.clip(d_ey * 10.0, -1.0, 1.0)
        # d_alpha is ~0.0001/step, need much more amplification
        d_alpha_norm = np.clip(d_alpha_norm * 100.0, -1.0, 1.0)

        return np.array([
            np.clip(ex, -1.0, 1.0),
            np.clip(ey, -1.0, 1.0),
            np.clip(alpha_norm, 0.0, 1.0),
            d_ex, d_ey, d_alpha_norm
        ], dtype=np.float32)

    def _action_to_real(self, action_01):
        alpha_star = self.ALPHA_MIN_REAL + action_01[0] * \
                     (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)
        lam = float(action_01[1])
        return alpha_star, lam

    def _get_ideal_action_01(self, alpha, d_ex, d_ey, d_alpha_norm):
        """
        Ideal mapping — the "teacher signal" for the reward.

        PPO should learn:
          far → high alpha* (approach), moderate lambda
          hold → alpha* ≈ 0.007, moderate lambda
          close → low alpha* (retreat), high lambda
          fast target → increase alpha* AND lambda
        """
        alpha_c = np.clip(alpha, 0.001, 0.030)

        # Base alpha*: monotonic decrease with alpha
        ratio = (alpha_c - 0.001) / 0.029
        ideal_alpha_star = 0.012 - ratio * 0.008
        ideal_alpha_star = np.clip(ideal_alpha_star, 0.003, 0.015)

        # Fast-target adjustment: increase alpha* (follow closer)
        rate_mag = math.sqrt(d_ex**2 + d_ey**2)
        ideal_alpha_star += rate_mag * 0.004
        ideal_alpha_star = np.clip(ideal_alpha_star, 0.003, 0.015)

        ideal_alpha_norm = (ideal_alpha_star - self.ALPHA_MIN_REAL) / \
                           (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)

        # Lambda: base on distance + rate boost
        dist = abs(alpha_c - self.TARGET_ALPHA)
        ideal_lambda = 0.35 + (dist / 0.010) * 0.30
        ideal_lambda = np.clip(ideal_lambda, 0.3, 0.85)

        # High rates → trust controller more
        ideal_lambda += rate_mag * 0.25
        ideal_lambda = np.clip(ideal_lambda, 0.3, 0.95)

        return np.clip(ideal_alpha_norm, 0.0, 1.0), np.clip(ideal_lambda, 0.0, 1.0)

    def _resample_target_motion(self, rng, mp):
        if self._step_count >= self._next_omega_step:
            self._target_omega = rng.uniform(-mp['omega_max'], mp['omega_max'])
            interval = rng.integers(mp['resample_min'], mp['resample_max'])
            self._next_omega_step = self._step_count + interval

        if self._step_count >= self._next_speed_step:
            self._target_speed = rng.uniform(mp['speed_min'], mp['speed_max'])
            interval = rng.integers(mp['resample_min'], mp['resample_max'])
            self._next_speed_step = self._step_count + interval

        if self._step_count >= self._next_vz_step:
            self._target_vz = rng.uniform(mp['vz_min'], mp['vz_max'])
            interval = rng.integers(mp['resample_min'], mp['resample_max'])
            self._next_vz_step = self._step_count + interval

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        ex = rng.uniform(-0.3, 0.3)
        ey = rng.uniform(-0.3, 0.3)

        # Balanced regime coverage
        regime = rng.integers(0, 4)
        if regime == 0:
            alpha = rng.uniform(0.001, 0.004)
        elif regime == 1:
            alpha = rng.uniform(0.004, 0.008)
        elif regime == 2:
            alpha = rng.uniform(0.006, 0.010)
        else:
            alpha = rng.uniform(0.010, 0.025)

        self._state_real = np.array([ex, ey, alpha], dtype=np.float32)
        self._prev_state_real = None
        self._step_count = 0
        self._dropout_count = 0

        mp = self.motion_params[self.difficulty]
        self._target_heading = rng.uniform(0, 2 * np.pi)
        self._target_omega = 0.0
        self._target_speed = rng.uniform(mp['speed_min'], mp['speed_max'])
        self._target_vz = 0.0
        self._next_omega_step = 0
        self._next_speed_step = 0
        self._next_vz_step = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step_count += 1

        self._prev_state_real = self._state_real.copy()
        ex, ey, alpha = self._state_real

        action_01 = (action + 1.0) / 2.0
        alpha_star, lam = self._action_to_real(action_01)

        # Get rates for reward
        obs_for_reward = self._get_obs()
        d_ex_s = obs_for_reward[3]
        d_ey_s = obs_for_reward[4]
        d_alpha_s = obs_for_reward[5]

        # ── REWARD ───────────────────────────────────────────────────
        ideal_alpha_norm, ideal_lambda = self._get_ideal_action_01(
            alpha, d_ex_s / 10.0, d_ey_s / 10.0, d_alpha_s / 100.0)

        alpha_sim  = np.exp(-((action_01[0] - ideal_alpha_norm)**2) / (2 * 0.12**2))
        lambda_sim = np.exp(-((action_01[1] - ideal_lambda)**2) / (2 * 0.08**2))
        centering  = np.exp(-(ex**2 + ey**2) / (2 * 0.20**2))

        dist_to_target = abs(alpha - self.TARGET_ALPHA)
        distance_sim = np.exp(-(dist_to_target**2) / (2 * 0.004**2))

        raw_rate = math.sqrt((d_ex_s/10.0)**2 + (d_ey_s/10.0)**2)
        stability = np.exp(-(raw_rate**2) / (2 * 0.012**2))

        survival = 0.5

        reward = (
            2.0 * alpha_sim +
            3.5 * lambda_sim +
            1.0 * centering +
            1.5 * distance_sim +
            1.0 * stability +
            survival
        )

        # ── IBVS v6.15 physics ───────────────────────────────────────
        gain = 0.3 + 0.7 * lam
        err_a = alpha - alpha_star

        if err_a < 0:
            vx = self.K_FAR * np.sqrt(-err_a) * gain
        else:
            vx = -self.K_NEAR * np.sqrt(err_a) * gain
        vx = np.clip(vx, -self.MAX_VX_RETREAT, self.MAX_VX)

        vy = -gain * self.Kp_y * ex
        vz = -gain * self.Kp_z * ey
        vy = np.clip(vy, -self.MAX_VY, self.MAX_VY)
        vz = np.clip(vz, -self.MAX_VZ, self.MAX_VZ)

        noise = 0.002
        new_ex    = ex    - vy * self.DT * self.VY_TO_EX_SCALE + \
                    self.np_random.normal(0, noise)
        new_ey    = ey    - vz * self.DT * self.VZ_TO_EY_SCALE + \
                    self.np_random.normal(0, noise)
        new_alpha = alpha + vx * self.DT * self.VX_TO_ALPHA_SCALE + \
                    self.np_random.normal(0, noise * 0.005)

        # ── Target motion ────────────────────────────────────────────
        mp = self.motion_params[self.difficulty]
        if mp['speed_max'] > 0:
            self._resample_target_motion(self.np_random, mp)
            self._target_heading += self._target_omega

            new_ex    += self._target_speed * math.cos(self._target_heading)
            new_ey    += self._target_speed * math.sin(self._target_heading)
            new_alpha += self._target_vz

        new_ex    = np.clip(new_ex,    -1.0,   1.0)
        new_ey    = np.clip(new_ey,    -1.0,   1.0)
        new_alpha = np.clip(new_alpha,  0.0003, 0.050)
        self._state_real = np.array([new_ex, new_ey, new_alpha], dtype=np.float32)

        target_lost = (
            abs(new_ex) > 0.95 or abs(new_ey) > 0.95 or
            new_alpha < 0.0005 or new_alpha > 0.045
        )
        if target_lost:
            self._dropout_count += 1
        else:
            self._dropout_count = 0

        terminated = self._dropout_count >= self.dropout_limit
        truncated  = self._step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {}


# =================================================================
#  CALLBACKS
# =================================================================
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


class ActionMonitorCallback(BaseCallback):
    def __init__(self, check_every=100_000, verbose=1):
        super().__init__(verbose)
        self.check_every = check_every
        self.recent_actions = deque(maxlen=500)

    def _on_step(self) -> bool:
        if self.locals.get("actions") is not None:
            for a in self.locals["actions"]:
                self.recent_actions.append(a.copy())

        if self.num_timesteps % self.check_every == 0 and \
           len(self.recent_actions) > 100:
            arr = np.array(list(self.recent_actions))
            arr_01 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
            print(f"\n  [Actions @ {self.num_timesteps:,}] "
                  f"alpha_norm: mean={np.mean(arr_01[:,0]):.3f} "
                  f"var={np.var(arr_01[:,0]):.4f} | "
                  f"lambda: mean={np.mean(arr_01[:,1]):.3f} "
                  f"var={np.var(arr_01[:,1]):.4f}")
        return True


# =================================================================
#  HELPERS
# =================================================================
def make_env(difficulty):
    def _init():
        return Monitor(DroneTrackingEnv(difficulty=difficulty))
    return _init


def evaluate_model(model, difficulty='medium', n_episodes=5):
    env = DroneTrackingEnv(difficulty=difficulty)
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
        final_alpha_real = obs[2] * DroneTrackingEnv.ALPHA_OBS_MAX
        print(f"  Eval ep {ep+1}: reward={ep_reward:.1f}  steps={step+1}  "
              f"final_alpha={final_alpha_real:.4f}")
    mean_r = np.mean(total_rewards)
    print(f"  Mean reward ({difficulty}): {mean_r:.1f}")
    return mean_r


def verify_state_dependence(model):
    test_cases = [
        ("VERY FAR  a=0.002 static",  [0.0, 0.0, 0.002, 0.0, 0.0, 0.0]),
        ("FAR       a=0.004 static",  [0.0, 0.0, 0.004, 0.0, 0.0, 0.0]),
        ("APPROACH  a=0.006 static",  [0.0, 0.0, 0.006, 0.0, 0.0, 0.0]),
        ("HOLD      a=0.007 static",  [0.0, 0.0, 0.007, 0.0, 0.0, 0.0]),
        ("CLOSE     a=0.012 static",  [0.0, 0.0, 0.012, 0.0, 0.0, 0.0]),
        ("V.CLOSE   a=0.018 static",  [0.0, 0.0, 0.018, 0.0, 0.0, 0.0]),
        ("HOLD a=0.007 fast_lateral",  [0.0, 0.0, 0.007, 0.3, 0.2, 0.0]),
        ("HOLD a=0.007 fast_approach", [0.0, 0.0, 0.007, 0.0, 0.0, 0.4]),
        ("FAR  a=0.004 fast_lateral",  [0.1, 0.0, 0.004, 0.4, 0.0, 0.0]),
    ]

    print("\n" + "=" * 85)
    print("STATE-DEPENDENCE VERIFICATION (v5.1 — calibrated physics)")
    print("=" * 85)
    print(f"  {'State':<30} | alpha*  (ideal)   lambda (ideal)")
    print("  " + "-" * 78)

    env_ref = DroneTrackingEnv()
    actions_01 = []
    for desc, vals in test_cases:
        ex, ey, alpha_real = vals[0], vals[1], vals[2]
        d_ex_s, d_ey_s, d_alpha_s = vals[3], vals[4], vals[5]

        alpha_norm_obs = alpha_real / DroneTrackingEnv.ALPHA_OBS_MAX
        obs = np.array([
            ex, ey, alpha_norm_obs,
            np.clip(d_ex_s, -1.0, 1.0),
            np.clip(d_ey_s, -1.0, 1.0),
            np.clip(d_alpha_s, -1.0, 1.0),
        ], dtype=np.float32)

        act, _ = model.predict(obs, deterministic=True)
        act_01 = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        alpha_real_out = DroneTrackingEnv.ALPHA_MIN_REAL + \
                         act_01[0] * (DroneTrackingEnv.ALPHA_MAX_REAL -
                                      DroneTrackingEnv.ALPHA_MIN_REAL)
        actions_01.append(act_01.copy())

        ideal_an, ideal_l = env_ref._get_ideal_action_01(
            alpha_real, d_ex_s/10.0, d_ey_s/10.0, d_alpha_s/100.0)
        ideal_ar = DroneTrackingEnv.ALPHA_MIN_REAL + \
                   ideal_an * (DroneTrackingEnv.ALPHA_MAX_REAL -
                               DroneTrackingEnv.ALPHA_MIN_REAL)
        print(f"  {desc:<30} | {alpha_real_out:.4f} ({ideal_ar:.4f})  "
              f"{act_01[1]:.3f} ({ideal_l:.3f})")

    arr = np.array(actions_01)
    alpha_var = np.var(arr[:, 0])
    lambda_var = np.var(arr[:, 1])
    print(f"\n  Action variance: alpha_norm={alpha_var:.4f}  "
          f"lambda={lambda_var:.4f}")

    passed = True

    far_a  = DroneTrackingEnv.ALPHA_MIN_REAL + actions_01[0][0] * \
             (DroneTrackingEnv.ALPHA_MAX_REAL - DroneTrackingEnv.ALPHA_MIN_REAL)
    hold_a = DroneTrackingEnv.ALPHA_MIN_REAL + actions_01[3][0] * \
             (DroneTrackingEnv.ALPHA_MAX_REAL - DroneTrackingEnv.ALPHA_MIN_REAL)
    close_a = DroneTrackingEnv.ALPHA_MIN_REAL + actions_01[5][0] * \
              (DroneTrackingEnv.ALPHA_MAX_REAL - DroneTrackingEnv.ALPHA_MIN_REAL)

    print()
    if far_a > close_a + 0.001:
        print(f"  ✓ Far alpha*({far_a:.4f}) > Close alpha*({close_a:.4f})")
    else:
        print(f"  ✗ Far alpha*({far_a:.4f}) NOT > Close alpha*({close_a:.4f})")
        passed = False

    if 0.005 < hold_a < 0.012:
        print(f"  ✓ Hold alpha*={hold_a:.4f} (in target range)")
    else:
        print(f"  ✗ Hold alpha*={hold_a:.4f} (should be 0.005–0.012)")
        passed = False

    if alpha_var > 0.002:
        print(f"  ✓ Alpha* variance={alpha_var:.4f}")
    else:
        print(f"  ✗ Alpha* variance={alpha_var:.4f} (too low)")
        passed = False

    if lambda_var > 0.002:
        print(f"  ✓ Lambda variance={lambda_var:.4f}")
    else:
        print(f"  ✗ Lambda variance={lambda_var:.4f} (collapsed)")
        passed = False

    static_hold_lam = actions_01[3][1]
    fast_hold_lam = actions_01[6][1]
    if fast_hold_lam > static_hold_lam + 0.02:
        print(f"  ✓ Fast-target lambda({fast_hold_lam:.3f}) > "
              f"Static lambda({static_hold_lam:.3f})  [rate-responsive]")
    else:
        print(f"  ⚠ Fast-target lambda({fast_hold_lam:.3f}) vs "
              f"Static lambda({static_hold_lam:.3f})  [weak rate response]")

    fast_approach_a = DroneTrackingEnv.ALPHA_MIN_REAL + actions_01[7][0] * \
                      (DroneTrackingEnv.ALPHA_MAX_REAL - DroneTrackingEnv.ALPHA_MIN_REAL)
    if fast_approach_a > hold_a:
        print(f"  ✓ Fast-approach alpha*({fast_approach_a:.4f}) > "
              f"Static hold alpha*({hold_a:.4f})")
    else:
        print(f"  ⚠ Fast-approach alpha*({fast_approach_a:.4f}) vs "
              f"Static hold({hold_a:.4f})")

    lam_mean = np.mean(arr[:, 1])
    if lam_mean > 0.20:
        print(f"  ✓ Lambda mean={lam_mean:.3f} (not collapsed)")
    else:
        print(f"  ✗ Lambda mean={lam_mean:.3f} (collapsed to 0)")
        passed = False

    print(f"\n  VERDICT: {'✓ PASS' if passed else '✗ NEEDS INVESTIGATION'}")
    return passed


# =================================================================
#  MAIN
# =================================================================
if __name__ == "__main__":
    N_ENVS     = 8
    BATCH_SIZE = 256
    N_STEPS    = 512
    N_EPOCHS   = 10
    LR         = 3e-4
    GAMMA      = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF   = 0.008
    VF_COEF    = 0.5
    MAX_GRAD_NORM = 0.5
    NET_ARCH   = [256, 256]
    LOG_STD_INIT = -0.5

    PHASE1_STEPS = 800_000
    PHASE2_STEPS = 800_000
    PHASE3_STEPS = 1_000_000
    PHASE4_STEPS = 800_000

    TOTAL = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS + PHASE4_STEPS

    print("=" * 70)
    print("PPO v5.1 — Calibrated Physics + 6D Observations")
    print("=" * 70)
    print(f"  Obs: 6D [ex, ey, alpha_n, d_ex*10, d_ey*10, d_alpha_n*100]")
    print(f"  Act: 2D [alpha_norm, lambda_norm]")
    print(f"  TARGET_ALPHA: {DroneTrackingEnv.TARGET_ALPHA}")
    print(f"  ALPHA range: [{DroneTrackingEnv.ALPHA_MIN_REAL}, "
          f"{DroneTrackingEnv.ALPHA_MAX_REAL}]")
    print(f"  Physics: VY_EX={DroneTrackingEnv.VY_TO_EX_SCALE:.3f} "
          f"VX_ALPHA={DroneTrackingEnv.VX_TO_ALPHA_SCALE:.5f}")
    print(f"  Net: {NET_ARCH}  |  Envs: {N_ENVS}  |  Steps: {TOTAL:,}")
    print(f"  Episodes: 500 steps (25 sec at 20Hz)")
    print()

    policy_kwargs = dict(net_arch=NET_ARCH, log_std_init=LOG_STD_INIT)
    DEVICE = 'cpu'

    # ── PHASE 1 ──────────────────────────────────────────────────────
    print("=" * 70)
    print(f"PHASE 1 — Easy ({PHASE1_STEPS:,} steps)")
    print("=" * 70)
    env1 = DummyVecEnv([make_env('easy') for _ in range(N_ENVS)])
    model = PPO(
        'MlpPolicy', env1,
        learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENT_COEF, vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs, verbose=1, device=DEVICE,
    )
    cb1 = RewardLoggerCallback()
    ac1 = ActionMonitorCallback(check_every=200_000)
    model.learn(total_timesteps=PHASE1_STEPS, callback=[cb1, ac1],
                progress_bar=True)
    env1.close()
    print(f"\n✓ Phase 1 done — {len(cb1.episode_rewards)} episodes")
    if cb1.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb1.episode_rewards[-50:]):.2f}")
    evaluate_model(model, 'easy', 3)
    gc.collect()

    # ── PHASE 2 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE 2 — Medium ({PHASE2_STEPS:,} steps)")
    print("=" * 70)
    env2 = DummyVecEnv([make_env('medium') for _ in range(N_ENVS)])
    model.set_env(env2)
    cb2 = RewardLoggerCallback()
    ac2 = ActionMonitorCallback(check_every=200_000)
    model.learn(total_timesteps=PHASE2_STEPS, callback=[cb2, ac2],
                reset_num_timesteps=False, progress_bar=True)
    env2.close()
    print(f"\n✓ Phase 2 done — {len(cb2.episode_rewards)} episodes")
    if cb2.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb2.episode_rewards[-50:]):.2f}")
    evaluate_model(model, 'medium', 3)
    gc.collect()

    # ── PHASE 3 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE 3 — Hard ({PHASE3_STEPS:,} steps)")
    print("=" * 70)
    env3 = DummyVecEnv([make_env('hard') for _ in range(N_ENVS)])
    model.set_env(env3)
    cb3 = RewardLoggerCallback()
    ac3 = ActionMonitorCallback(check_every=200_000)
    model.learn(total_timesteps=PHASE3_STEPS, callback=[cb3, ac3],
                reset_num_timesteps=False, progress_bar=True)
    env3.close()
    print(f"\n✓ Phase 3 done — {len(cb3.episode_rewards)} episodes")
    if cb3.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb3.episode_rewards[-50:]):.2f}")
    evaluate_model(model, 'hard', 3)
    gc.collect()

    # ── PHASE 4 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE 4 — Extreme ({PHASE4_STEPS:,} steps)")
    print("=" * 70)
    env4 = DummyVecEnv([make_env('extreme') for _ in range(N_ENVS)])
    model.set_env(env4)
    cb4 = RewardLoggerCallback()
    ac4 = ActionMonitorCallback(check_every=200_000)
    model.learn(total_timesteps=PHASE4_STEPS, callback=[cb4, ac4],
                reset_num_timesteps=False, progress_bar=True)
    env4.close()
    print(f"\n✓ Phase 4 done — {len(cb4.episode_rewards)} episodes")
    if cb4.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb4.episode_rewards[-50:]):.2f}")

    # ── Final evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ALL PHASES COMPLETE — {TOTAL:,} steps")
    print("=" * 70)

    print("\nFinal evaluation:")
    for d in ['easy', 'medium', 'hard', 'extreme']:
        print(f"\n  --- {d.upper()} ---")
        evaluate_model(model, d, 5)

    passed = verify_state_dependence(model)

    WEIGHTS = os.path.expanduser(
        "~/drone_detection/models/ppo_policy_weights_v5.pth")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save(model.policy.state_dict(), WEIGHTS)
    print(f"\n✓ Weights saved: {WEIGHTS}")
    print(f"  Size: {os.path.getsize(WEIGHTS) / 1024:.1f} KB")

    print("\n  v5.1 observation scaling (MUST match ROS node):")
    print("    d_ex  amplification: 10.0")
    print("    d_ey  amplification: 10.0")
    print("    d_alpha_norm amplification: 100.0")
    print(f"  Architecture: obs=6 → {NET_ARCH} → act=2")
    print(f"  Alpha range: [{DroneTrackingEnv.ALPHA_MIN_REAL}, "
          f"{DroneTrackingEnv.ALPHA_MAX_REAL}]")

    model.save(os.path.expanduser(
        "~/drone_detection/models/ppo_full_model_v5"))
    print("  Full model backup: ppo_full_model_v5.zip")

    print("\n" + "=" * 70)
    print(f"  VERDICT: {'✓ PASS' if passed else '✗ NEEDS INVESTIGATION'}")
    print(f"  Weights at: {WEIGHTS}")
    print("=" * 70)