#!/usr/bin/env python3
"""bc_to_td3_surgery.py — merge a trained BC policy into a fresh TD3 actor (offline).

The BC net (BCPolicy: obs->256->256->4, tanh) has dims IDENTICAL to the SB3 TD3
actor.mu (net_arch=[256,256], squash_output=True). So warm-starting = copy the three
Linear layers BC.net.{0,2,4} -> TD3 actor.mu.{0,2,4}, then sync actor_target<-actor.
TD3 is deterministic -> no log_std head to tame (unlike the SAC surgery in rl_train_sac).

No ROS/Gazebo needed: we instantiate TD3 with a dummy Box(14)/Box(4) env just to get the
network shells, then overwrite the actor weights and save. The fine-tune run loads this
zip (as CAPSTD3 or TD3 — same architecture) and continues online RL from the BC behaviour.

Usage: python3 bc_to_td3_surgery.py --bc ~/fyp/rl/models/bc_policy_v5.pth \
                                     --out ~/fyp/rl/models/model_B_td3_warmstart.zip
"""
import os, argparse, numpy as np, torch
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

OBS_DIM, ACT_DIM = 14, 4

class _Stub(gym.Env):
    """Minimal env: correct spaces only (never stepped)."""
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    def reset(self, *a, **k): return np.zeros(OBS_DIM, np.float32), {}
    def step(self, a): return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc',  default=os.path.expanduser('~/fyp/rl/models/bc_policy_v5.pth'))
    ap.add_argument('--out', default=os.path.expanduser('~/fyp/rl/models/model_B_td3_warmstart.zip'))
    a = ap.parse_args()
    bc_path = os.path.expanduser(a.bc); out_path = os.path.expanduser(a.out)

    ck = torch.load(bc_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    bc_obs = ck.get('obs_dim', 'NA'); caps = ck.get('caps', 'NA')
    print(f"[surgery] BC: {bc_path}  obs_dim={bc_obs} caps={caps} val_mse={ck.get('val_mse')}")
    assert sd['net.0.weight'].shape == (256, OBS_DIM), f"BC input {sd['net.0.weight'].shape} != (256,{OBS_DIM})"

    venv = DummyVecEnv([lambda: _Stub()])
    model = TD3("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                learning_starts=0, buffer_size=1, verbose=0, device='cpu')

    actor = model.policy.actor
    actor_t = model.policy.actor_target
    # show the actual mu layer layout so the mapping is verified, not assumed
    print("[surgery] TD3 actor.mu:", [f"{i}:{type(l).__name__}" for i,l in enumerate(actor.mu)])
    with torch.no_grad():
        actor.mu[0].weight.copy_(sd['net.0.weight']); actor.mu[0].bias.copy_(sd['net.0.bias'])
        actor.mu[2].weight.copy_(sd['net.2.weight']); actor.mu[2].bias.copy_(sd['net.2.bias'])
        actor.mu[4].weight.copy_(sd['net.4.weight']); actor.mu[4].bias.copy_(sd['net.4.bias'])
        actor_t.load_state_dict(actor.state_dict())   # target <- actor
    model.save(out_path)
    print(f"[surgery] saved warm-started TD3 -> {out_path}")

    # ---- SELF-TEST: reload + confirm TD3 actor == BC policy on random obs ----
    from rl_train_bc import BCPolicy
    bc = BCPolicy(obs_dim=OBS_DIM); bc.load_state_dict(sd); bc.eval()
    m2 = TD3.load(out_path, device='cpu')
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.5, size=(200, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        a_bc = bc(torch.from_numpy(X)).numpy()
    a_td3 = np.array([m2.predict(x, deterministic=True)[0] for x in X])
    err = np.abs(a_bc - a_td3).max()
    print(f"\n[SELF-TEST] max|BC action - TD3 action| over 200 random obs = {err:.2e}")
    print(f"[SELF-TEST] {'PASS — surgery correct (TD3 reproduces BC)' if err < 1e-4 else 'FAIL — weights not matching!'}")
    # behaviour sanity: far centered target -> should pursue (vx>0)
    far = np.zeros(OBS_DIM, np.float32); far[2] = 1.5
    print(f"[SANITY] far-centered target action [vx,vy,vz,wz] = {np.round(m2.predict(far, deterministic=True)[0],3)}  (vx>0 = pursues = IBVS-like)")

if __name__ == '__main__':
    main()
