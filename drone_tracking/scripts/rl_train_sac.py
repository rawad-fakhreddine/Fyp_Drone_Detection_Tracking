#!/usr/bin/env python3
"""
rl_train_sac.py — RL Milestone (Config 3) · online SAC training (SKELETON).

STATUS: wiring only — NOT runnable until Step 3 (reset) and Step 4 (reward) land
in rl_env.DroneTrackingEnv. Everything here is the locked design of
FYP/RL/training/Training_Design.docx so the online phase is a fill-in, not a design.

Plan (locked):
  - SAC (primary; off-policy replay suits the ~real-time Gazebo sim), PPO baseline.
  - Warm-start: load bc_policy_v1.pth weights into the SAC actor body before
    any environment interaction (BC = starting line, not the ceiling).
  - VecFrameStack(N=4) over the 16-dim single-frame obs -> 64-dim policy input
    (ordering matches rl_bc_dataset.stack(): oldest first).
  - Episodes: Option A — fixed ~30-40 s (truncation, bootstrapped) OR terminal on
    collision (-P_safe) / sustained loss > 3 s (-P_lost_final, largest penalty).
  - Reset: try RESET-FREE first (re-randomize the TARGET trajectory relative to the
    chaser's current pose — no teleport, no EKF risk), teleport+settle as fallback.
  - Curriculum: T1/T2 -> T3/T4 -> T5/T7 (avoid local optima).
  - Reward: FYP/RL/reward/Reward_Design.docx (centering + band + smoothness +
    keep-in-view + collision; GT allowed — training only).

Usage (once Steps 3-4 are in):
  rosrun drone_tracking rl_train_sac.py _bc:=~/fyp/rl/models/bc_policy_v1.pth
"""
import os, sys
import numpy as np

def build(bc_path=None, total_steps=200_000):
    import rospy, torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from rl_env import DroneTrackingEnv

    env = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv()]), n_stack=4)
    model = SAC("MlpPolicy", env,
                learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                train_freq=1, gradient_steps=1, tau=0.005, gamma=0.99,
                policy_kwargs=dict(net_arch=[256, 256]),   # matches BCPolicy dims
                verbose=1, tensorboard_log=os.path.expanduser('~/fyp/rl/logs'))

    if bc_path and os.path.exists(os.path.expanduser(bc_path)):
        ck = torch.load(os.path.expanduser(bc_path), map_location='cpu')
        sd = ck['state_dict']
        actor = model.policy.actor
        with torch.no_grad():   # BC body (64->256->256) -> SAC actor latent_pi
            actor.latent_pi[0].weight.copy_(sd['net.0.weight']); actor.latent_pi[0].bias.copy_(sd['net.0.bias'])
            actor.latent_pi[2].weight.copy_(sd['net.2.weight']); actor.latent_pi[2].bias.copy_(sd['net.2.bias'])
            actor.mu.weight.copy_(sd['net.4.weight']);           actor.mu.bias.copy_(sd['net.4.bias'])
        print(f"[sac] warm-started actor from {bc_path} (val MSE {ck.get('val_mse'):.5f})")
    return model

if __name__ == '__main__':
    print(__doc__)
    print("NOT RUNNABLE YET: rl_env.DroneTrackingEnv needs Step 3 (reset) + Step 4 "
          "(reward/termination) before SAC.learn() can be called.")
