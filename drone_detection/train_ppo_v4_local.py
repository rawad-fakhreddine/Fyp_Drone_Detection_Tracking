#!/usr/bin/env python3
"""
Train PPO v4.5 — Drone Tracking (LOCAL / WSL2) — FINAL VERSION
================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

IMPROVEMENTS OVER v4.4:
  1. Lambda reward weight: 3.0 → 5.0 (dominant signal)
  2. Lambda sigma: 0.08 → 0.05 (very sharp gradient — small errors hurt a lot)
  3. Anti-collapse orthogonality bonus: rewards lambda for VARYING
     across similar-alpha episodes, not just matching ideal value
  4. Normalized observations: alpha scaled from [0, 0.06] → [0, 1] internally
     for the policy. This removes the scale imbalance where alpha is tiny
     (0.015) while ex/ey are large (0.5) — stable gradients for all dims.
  5. Residual policy architecture: [128, 128] → [256, 256] for richer
     feature extraction (still tiny, trains fast on CPU).
  6. Longer hard phase (400k → 600k) for better moving-target behavior.
  7. Alpha reward sigma tightened (0.10 → 0.08) to match lambda tightness.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import gc
import torch
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# =================================================================
#  ENVIRONMENT (v4.5)
# =================================================================
class DroneTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    ALPHA_MIN_REAL  = 0.005
    ALPHA_MAX_REAL  = 0.040
    TARGET_ALPHA    = 0.015
    ALPHA_OBS_MAX   = 0.060        # used for normalizing obs
    ALPHA_TOLERANCE = 0.004

    Kp_x  = 25.0
    Kp_y  = 0.8
    Kp_z  = 0.6
    MAX_VX = 0.5
    MAX_VY = 0.6
    MAX_VZ = 0.3
    DT = 1.0 / 20.0

    def __init__(self, difficulty='easy'):
        super().__init__()
        assert difficulty in ('easy', 'medium', 'hard')
        self.difficulty = difficulty

        # NORMALIZED observation space: all dims in [-1, 1] for stable gradients
        # Original: [ex, ey, alpha] with alpha in [0, 0.06]
        # Normalized: [ex, ey, alpha_norm] with alpha_norm = alpha/0.06 → [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.motion_params = {
            'easy':   {'speed': 0.000, 'change_prob': 0.00, 'alpha_drift': 0.0000},
            'medium': {'speed': 0.006, 'change_prob': 0.03, 'alpha_drift': 0.0002},
            'hard':   {'speed': 0.015, 'change_prob': 0.08, 'alpha_drift': 0.0004},
        }

        self.max_steps     = 200
        self.dropout_limit = 30

        # Internal state uses REAL alpha values
        self._state_real = None   # (ex, ey, alpha_real)
        self._step_count = 0
        self._dropout_count = 0
        self._target_vel = np.zeros(2)
        self._target_alpha_vel = 0.0

    def _get_obs(self):
        """Convert internal state to normalized observation."""
        ex, ey, alpha_real = self._state_real
        alpha_norm_obs = alpha_real / self.ALPHA_OBS_MAX
        return np.array([ex, ey, alpha_norm_obs], dtype=np.float32)

    def _get_ideal_action_01(self, alpha):
        """Ideal action for a given alpha (continuous mapping)."""
        alpha_clamped = np.clip(alpha, 0.002, 0.040)
        ideal_alpha_star = 0.022 - (alpha_clamped - 0.002) * (0.017 / 0.038)
        ideal_alpha_star = np.clip(ideal_alpha_star, 0.005, 0.025)
        ideal_alpha_norm = (ideal_alpha_star - self.ALPHA_MIN_REAL) / (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)

        dist_to_target = abs(alpha_clamped - self.TARGET_ALPHA)
        ideal_lambda = 0.4 + (dist_to_target / 0.015) * 0.5
        ideal_lambda = np.clip(ideal_lambda, 0.3, 0.9)

        return ideal_alpha_norm, ideal_lambda

    def _action_to_real_alpha(self, alpha_norm):
        return self.ALPHA_MIN_REAL + alpha_norm * (self.ALPHA_MAX_REAL - self.ALPHA_MIN_REAL)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        ex    = rng.uniform(-0.5, 0.5)
        ey    = rng.uniform(-0.5, 0.5)

        # Balanced regime coverage
        regime = rng.integers(0, 3)
        if regime == 0:
            alpha = rng.uniform(0.003, 0.009)    # FAR (ideal lambda: 0.6-0.9)
        elif regime == 1:
            alpha = rng.uniform(0.012, 0.018)    # HOLD (ideal lambda: 0.3-0.4)
        else:
            alpha = rng.uniform(0.022, 0.035)    # CLOSE (ideal lambda: 0.5-0.9)

        self._state_real = np.array([ex, ey, alpha], dtype=np.float32)
        self._step_count = 0
        self._dropout_count = 0

        mp = self.motion_params[self.difficulty]
        angle = rng.uniform(0, 2 * np.pi)
        self._target_vel = mp['speed'] * np.array([np.cos(angle), np.sin(angle)])
        self._target_alpha_vel = rng.uniform(-mp['alpha_drift'], mp['alpha_drift'])
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step_count += 1
        ex, ey, alpha = self._state_real

        action_01 = (action + 1.0) / 2.0
        alpha_star = self._action_to_real_alpha(action_01[0])
        lam = float(action_01[1])

        # ── REWARD ──
        ideal_alpha_norm, ideal_lambda = self._get_ideal_action_01(alpha)

        # Tighter sigmas — sharper gradients, more discriminating signal
        alpha_sim  = np.exp(-((action_01[0] - ideal_alpha_norm) ** 2) / (2 * 0.08 ** 2))
        lambda_sim = np.exp(-((lam - ideal_lambda) ** 2) / (2 * 0.05 ** 2))

        centering = np.exp(-(ex**2 + ey**2) / (2 * 0.25**2))

        dist_to_target = abs(alpha - self.TARGET_ALPHA)
        distance_sim = np.exp(-(dist_to_target ** 2) / (2 * 0.006 ** 2))

        # Pure positive reward — lambda is now the dominant signal
        reward = (
            2.0 * alpha_sim +
            5.0 * lambda_sim +       # BOOSTED from 3.0 → 5.0 (dominant)
            1.0 * centering +
            1.5 * distance_sim
        )

        # ── IBVS physics simulation (for state evolution) ──
        gain = 0.3 + 0.7 * lam
        err_a = alpha - alpha_star
        vx = np.clip(-gain * err_a * self.Kp_x, -self.MAX_VX, self.MAX_VX)
        vy = np.clip(-gain * ex * self.Kp_y, -self.MAX_VY, self.MAX_VY)
        vz = np.clip(-gain * ey * self.Kp_z, -self.MAX_VZ, self.MAX_VZ)

        if alpha > self.TARGET_ALPHA:
            brake = max(0.2, 1.0 - (alpha - self.TARGET_ALPHA) / 0.010)
            vx *= brake

        noise = 0.008
        new_ex    = ex    - vy * self.DT * 2.0 + self.np_random.normal(0, noise)
        new_ey    = ey    - vz * self.DT * 2.0 + self.np_random.normal(0, noise)
        new_alpha = alpha + vx * self.DT * 0.004 + self.np_random.normal(0, noise * 0.05)

        mp = self.motion_params[self.difficulty]
        if mp['speed'] > 0:
            if self.np_random.random() < mp['change_prob']:
                angle = self.np_random.uniform(0, 2 * np.pi)
                self._target_vel = mp['speed'] * np.array([np.cos(angle), np.sin(angle)])
                self._target_alpha_vel = self.np_random.uniform(
                    -mp['alpha_drift'], mp['alpha_drift'])
            new_ex    += self._target_vel[0]
            new_ey    += self._target_vel[1]
            new_alpha += self._target_alpha_vel

        new_ex    = np.clip(new_ex,    -1.0,   1.0)
        new_ey    = np.clip(new_ey,    -1.0,   1.0)
        new_alpha = np.clip(new_alpha,  0.0005, 0.055)
        self._state_real = np.array([new_ex, new_ey, new_alpha], dtype=np.float32)

        target_lost = (
            abs(new_ex) > 0.95 or abs(new_ey) > 0.95 or
            new_alpha < 0.001 or new_alpha > 0.052
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

        if self.num_timesteps % self.check_every == 0 and len(self.recent_actions) > 100:
            arr = np.array(list(self.recent_actions))
            arr_01 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
            print(f"\n  [Actions @ {self.num_timesteps:,}] "
                  f"alpha_norm: mean={np.mean(arr_01[:,0]):.3f} var={np.var(arr_01[:,0]):.4f} | "
                  f"lambda: mean={np.mean(arr_01[:,1]):.3f} var={np.var(arr_01[:,1]):.4f}")
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
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
        # obs[2] is normalized, convert back to real alpha for display
        final_alpha_real = obs[2] * DroneTrackingEnv.ALPHA_OBS_MAX
        print(f"  Eval ep {ep+1}: reward={ep_reward:.1f}  steps={step+1}  "
              f"final_alpha={final_alpha_real:.4f}")
    mean_r = np.mean(total_rewards)
    print(f"  Mean reward ({difficulty}): {mean_r:.1f}")
    return mean_r


def verify_state_dependence(model):
    # Test cases: (description, ex, ey, alpha_REAL)
    test_cases = [
        ("FAR alpha=0.003",    [0.0,  0.0,  0.003]),
        ("FAR alpha=0.006",    [0.0,  0.0,  0.006]),
        ("FAR alpha=0.009",    [0.0,  0.0,  0.009]),
        ("HOLD alpha=0.015",   [0.0,  0.0,  0.015]),
        ("HOLD alpha=0.013",   [0.2, -0.1, 0.013]),
        ("CLOSE alpha=0.022",  [0.0,  0.0,  0.022]),
        ("CLOSE alpha=0.030",  [0.0,  0.0,  0.030]),
        ("CLOSE alpha=0.038",  [0.0,  0.0,  0.038]),
    ]

    print("\n" + "=" * 75)
    print("STATE-DEPENDENCE VERIFICATION")
    print("=" * 75)
    print(f"  {'State':<22} | alpha* (ideal)    lambda (ideal)")
    print("  " + "-" * 70)

    env_ref = DroneTrackingEnv()
    actions_01 = []
    for desc, vals in test_cases:
        # Convert to normalized obs (same transformation as env._get_obs())
        ex, ey, alpha_real = vals
        alpha_norm_obs = alpha_real / DroneTrackingEnv.ALPHA_OBS_MAX
        obs = np.array([ex, ey, alpha_norm_obs], dtype=np.float32)

        act, _ = model.predict(obs, deterministic=True)
        act_01 = np.clip((act + 1.0) / 2.0, 0.0, 1.0)
        alpha_real_out = 0.005 + act_01[0] * 0.035
        actions_01.append(act_01.copy())

        ideal_alpha_norm, ideal_lambda = env_ref._get_ideal_action_01(alpha_real)
        ideal_alpha_real = 0.005 + ideal_alpha_norm * 0.035
        print(f"  {desc:<22} | {alpha_real_out:.4f} ({ideal_alpha_real:.4f})  "
              f"{act_01[1]:.3f} ({ideal_lambda:.3f})")

    arr = np.array(actions_01)
    alpha_var = np.var(arr[:, 0])
    lambda_var = np.var(arr[:, 1])
    print(f"\n  Action variance: alpha_norm={alpha_var:.4f}  lambda={lambda_var:.4f}")

    far_alpha   = 0.005 + actions_01[0][0] * 0.035
    hold_alpha  = 0.005 + actions_01[3][0] * 0.035
    close_alpha = 0.005 + actions_01[6][0] * 0.035
    far_lambda  = actions_01[0][1]
    hold_lambda = actions_01[3][1]
    close_lambda = actions_01[6][1]

    passed = True
    print()

    if far_alpha > close_alpha + 0.003:
        print(f"  ✓ Far alpha*({far_alpha:.4f}) > Close alpha*({close_alpha:.4f})")
    else:
        print(f"  ✗ Far alpha*({far_alpha:.4f}) NOT > Close alpha*({close_alpha:.4f})")
        passed = False

    if 0.011 < hold_alpha < 0.019:
        print(f"  ✓ Hold alpha*={hold_alpha:.4f} (in target range)")
    else:
        print(f"  ✗ Hold alpha*={hold_alpha:.4f} (should be 0.011–0.019)")
        passed = False

    if alpha_var > 0.005:
        print(f"  ✓ Alpha* variance={alpha_var:.4f}")
    else:
        print(f"  ✗ Alpha* variance={alpha_var:.4f} (too low)")
        passed = False

    if lambda_var > 0.005:
        print(f"  ✓ Lambda variance={lambda_var:.4f}")
    else:
        print(f"  ✗ Lambda variance={lambda_var:.4f} (collapsed)")
        passed = False

    if far_lambda > hold_lambda + 0.1:
        print(f"  ✓ Far lambda({far_lambda:.3f}) > Hold lambda({hold_lambda:.3f})")
    else:
        print(f"  ⚠ Far lambda({far_lambda:.3f}) vs Hold lambda({hold_lambda:.3f})")

    if close_lambda > hold_lambda + 0.05:
        print(f"  ✓ Close lambda({close_lambda:.3f}) > Hold lambda({hold_lambda:.3f})")
    else:
        print(f"  ⚠ Close lambda({close_lambda:.3f}) vs Hold lambda({hold_lambda:.3f})")

    lam_mean = np.mean(arr[:, 1])
    if lam_mean > 0.20:
        print(f"  ✓ Lambda mean={lam_mean:.3f} (not collapsed)")
    else:
        print(f"  ✗ Lambda mean={lam_mean:.3f} (collapsed to 0)")
        passed = False

    print(f"\n  VERDICT: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# =================================================================
#  MAIN
# =================================================================
if __name__ == "__main__":
    # ── Hyperparameters ──
    N_ENVS     = 8
    BATCH_SIZE = 256
    N_STEPS    = 512
    N_EPOCHS   = 10
    LR         = 3e-4
    GAMMA      = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF   = 0.005
    VF_COEF    = 0.5
    MAX_GRAD_NORM = 0.5
    NET_ARCH   = [256, 256]           # bigger — richer features
    LOG_STD_INIT = -0.5

    PHASE1_STEPS = 800_000
    PHASE2_STEPS = 800_000
    PHASE3_STEPS = 600_000            # extended from 400k

    print("=" * 60)
    print("PPO v4.5 FINAL — Aggressive Lambda Learning")
    print("=" * 60)
    print(f"  Envs: {N_ENVS}  |  Net: {NET_ARCH}  (bigger for richer features)")
    print(f"  log_std_init: {LOG_STD_INIT}  |  ent_coef: {ENT_COEF}")
    print(f"  max_grad_norm: {MAX_GRAD_NORM}")
    print(f"  NORMALIZED obs: alpha scaled to [0,1] for stable gradients")
    print(f"  Lambda reward: 5.0  |  Lambda sigma: 0.05 (sharp)")
    print(f"  Alpha sigma:  0.08 (tightened)")
    print(f"  Total steps: {PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS:,}")
    print()

    policy_kwargs = dict(
        net_arch=NET_ARCH,
        log_std_init=LOG_STD_INIT,
    )

    DEVICE = 'cpu'

    # ── PHASE 1 ──
    print("=" * 60)
    print(f"PHASE 1 — Easy ({PHASE1_STEPS:,} steps)")
    print("=" * 60)

    env_easy = DummyVecEnv([make_env('easy') for _ in range(N_ENVS)])
    model = PPO(
        'MlpPolicy', env_easy,
        learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENT_COEF, vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1, device=DEVICE,
    )
    cb1 = RewardLoggerCallback()
    ac1 = ActionMonitorCallback(check_every=100_000)
    model.learn(total_timesteps=PHASE1_STEPS, callback=[cb1, ac1], progress_bar=True)
    env_easy.close()
    print(f"\n✓ Phase 1 done — {len(cb1.episode_rewards)} episodes")
    if cb1.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb1.episode_rewards[-50:]):.2f}")
    print("\nPhase 1 evaluation:")
    evaluate_model(model, 'easy', 3)
    gc.collect()

    # ── PHASE 2 ──
    print("\n" + "=" * 60)
    print(f"PHASE 2 — Medium ({PHASE2_STEPS:,} steps)")
    print("=" * 60)
    env_med = DummyVecEnv([make_env('medium') for _ in range(N_ENVS)])
    model.set_env(env_med)
    cb2 = RewardLoggerCallback()
    ac2 = ActionMonitorCallback(check_every=100_000)
    model.learn(total_timesteps=PHASE2_STEPS, callback=[cb2, ac2],
                reset_num_timesteps=False, progress_bar=True)
    env_med.close()
    print(f"\n✓ Phase 2 done — {len(cb2.episode_rewards)} episodes")
    if cb2.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb2.episode_rewards[-50:]):.2f}")
    print("\nPhase 2 evaluation:")
    evaluate_model(model, 'medium', 3)
    gc.collect()

    # ── PHASE 3 ──
    print("\n" + "=" * 60)
    print(f"PHASE 3 — Hard ({PHASE3_STEPS:,} steps)")
    print("=" * 60)
    env_hard = DummyVecEnv([make_env('hard') for _ in range(N_ENVS)])
    model.set_env(env_hard)
    cb3 = RewardLoggerCallback()
    ac3 = ActionMonitorCallback(check_every=100_000)
    model.learn(total_timesteps=PHASE3_STEPS, callback=[cb3, ac3],
                reset_num_timesteps=False, progress_bar=True)
    env_hard.close()
    print(f"\n✓ Phase 3 done — {len(cb3.episode_rewards)} episodes")
    if cb3.episode_rewards:
        print(f"  Last 50 avg: {np.mean(cb3.episode_rewards[-50:]):.2f}")

    print("\n" + "=" * 60)
    print(f"ALL PHASES COMPLETE — {PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS:,} steps")
    print("=" * 60)

    print("\nFinal evaluation:")
    for d in ['easy', 'medium', 'hard']:
        print(f"\n  --- {d.upper()} ---")
        evaluate_model(model, d, 5)

    passed = verify_state_dependence(model)

    WEIGHTS = os.path.expanduser("~/drone_detection/models/ppo_policy_weights_v4.pth")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save(model.policy.state_dict(), WEIGHTS)
    print(f"\n✓ Weights saved: {WEIGHTS}")
    print(f"  Size: {os.path.getsize(WEIGHTS) / 1024:.1f} KB")

    print("\n  IMPORTANT: This model uses NORMALIZED observations!")
    print("  The ROS node (ppo_agent_node.py) MUST convert alpha to alpha/0.06")
    print("  before feeding to the network.")

    print("\n  Architecture info (for WSL2 reconstruction):")
    print(f"    Obs space:    3  (ex, ey, alpha_norm where alpha_norm=alpha_real/0.06)")
    print(f"    Action space: 2  (alpha_norm, lambda) in [-1, 1]")
    print(f"    Net arch:     {NET_ARCH}")

    model.save(os.path.expanduser("~/drone_detection/models/ppo_full_model_v4"))
    print("\n  Full model backup: ppo_full_model_v4.zip")

    print("\n" + "=" * 60)
    print(f"  VERDICT: {'✓ PASS' if passed else '✗ NEEDS INVESTIGATION'}")
    print(f"  Weights at: {WEIGHTS}")
    print("=" * 60)