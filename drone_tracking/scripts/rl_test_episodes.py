#!/usr/bin/env python3
"""
rl_test_episodes.py — RL Milestone (Config 3) · Step 3 test: reset-free episodes.

Drives the gym DroneTrackingEnv through the SAME wrapper SAC will use
(DummyVecEnv + VecFrameStack N=4), with the BC policy as the actor, for N short
episodes. Validates that:
  - env.reset() / env.step() work through the SB3 VecEnv path,
  - episodes TRUNCATE at ~episode_secs (Option A time-limit),
  - reset is RESET-FREE (no teleport) — tracking continues across episode
    boundaries and the EKF stays stable.

Run AFTER handing control off from IBVS (start this so it is publishing, then
`rosnode kill /ibvs_controller_node`).

Usage:
  rosrun drone_tracking rl_test_episodes.py _episodes:=3 _episode_secs:=15 \
        _bc:=~/fyp/rl/models/bc_policy_v2.pth
"""
import os
import numpy as np
import rospy
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl_env import DroneTrackingEnv
from rl_train_bc import BCPolicy


def main():
    rospy.init_node('rl_test_episodes', anonymous=True)
    epsecs = float(rospy.get_param('~episode_secs', 15.0))
    nep    = int(rospy.get_param('~episodes', 3))
    bc     = os.path.expanduser(rospy.get_param('~bc', '~/fyp/rl/models/bc_policy_v2.pth'))

    ck = torch.load(bc, map_location='cpu', weights_only=False)
    net = BCPolicy(obs_dim=64); net.load_state_dict(ck['state_dict']); net.eval()

    env = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv(episode_secs=epsecs)]), n_stack=4)
    obs = env.reset()
    print(f"[test] first reset OK, stacked obs shape {obs.shape} (expect (1,64))", flush=True)

    ep = 0; steps = 0; dists = []; exs = []; eys = []; rews = []
    while not rospy.is_shutdown() and ep < nep:
        with torch.no_grad():
            a = net(torch.from_numpy(obs.astype(np.float32))).numpy()
        obs, rew, done, info = env.step(a)
        steps += 1
        inf = info[0]
        dists.append(inf.get('true_dist', float('nan')))
        exs.append(abs(float(obs[0, -16])))   # newest frame ex (last 16 dims, index 0)
        eys.append(abs(float(obs[0, -15])))   # newest frame ey
        rews.append(float(rew[0]))
        if steps % 20 == 0:
            print(f"  ep{ep} t{inf.get('elapsed', 0):4.1f}s  true_dist {inf.get('true_dist', 0):5.2f}"
                  f"  ex {obs[0, -16]:+.3f} ey {obs[0, -15]:+.3f}  r {float(rew[0]):+.3f}", flush=True)
        if done[0]:
            d = np.array(dists, dtype=float)
            trunc = inf.get('TimeLimit.truncated', False)
            print(f">>> EPISODE {ep} DONE ({len(dists)} steps, truncated={trunc}): "
                  f"mean_dist {np.nanmean(d):.2f} [min {np.nanmin(d):.2f}, max {np.nanmax(d):.2f}], "
                  f"mean|ex| {np.mean(exs):.3f} mean|ey| {np.mean(eys):.3f}, "
                  f"mean_reward {np.mean(rews):+.3f} (min {np.min(rews):+.3f}), sum {np.sum(rews):+.1f}",
                  flush=True)
            ep += 1; dists = []; exs = []; eys = []; rews = []
    print("[test] ALL EPISODES DONE — reset-free loop validated", flush=True)
    rospy.signal_shutdown("test done")


if __name__ == '__main__':
    main()
