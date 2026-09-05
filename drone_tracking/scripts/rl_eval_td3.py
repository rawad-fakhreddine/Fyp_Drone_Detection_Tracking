#!/usr/bin/env python3
"""
rl_eval_td3.py — RL Milestone (Config 3) · deterministic evaluation of a trained TD3
policy. Identical harness to rl_eval_sac.py (same env wrapper, reset-free, flight_logger
writes the standard 65-col ~/flight_log_latest.csv) — only the loader differs (TD3.load).

For TD3 deterministic=True is the SAME actor that was optimised during training (TD3's
actor is deterministic by construction) — so unlike SAC there is no mean-vs-sample gap
between what trained and what we score here.

Usage:
  rosrun drone_tracking rl_eval_td3.py _model:=~/fyp/rl/models/td3/td3_policy.zip _secs:=120
"""
import os
import numpy as np
import rospy
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl_env import DroneTrackingEnv


def main():
    rospy.init_node('rl_eval_td3', anonymous=True)
    model_p = os.path.expanduser(rospy.get_param('~model', '~/fyp/rl/models/td3/td3_policy.zip'))
    secs = float(rospy.get_param('~secs', 120.0))

    rospy.loginfo(f"[eval] loading TD3 policy: {model_p}")
    model = TD3.load(model_p, device='cpu')

    n_stack = int(rospy.get_param('~n_stack', 1))
    env = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv(episode_secs=1e9)]), n_stack=n_stack)
    obs = env.reset()
    print(f"[eval] reset OK, obs {obs.shape} (n_stack={n_stack}); deterministic run for {secs:.0f}s",
          flush=True)

    t0 = rospy.get_time()
    steps = 0
    dists, exs, eys = [], [], []
    print("TD3-EVAL-READY", flush=True)   # handoff marker: kill IBVS now
    while not rospy.is_shutdown() and (rospy.get_time() - t0) < secs:
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(a)
        inf = info[0]
        steps += 1
        dists.append(inf.get('true_dist', float('nan')))
        exs.append(abs(float(obs[0, -14]))); eys.append(abs(float(obs[0, -13])))
        if steps % 40 == 0:
            d = np.array(dists[-40:], dtype=float)
            print(f"TD3-EVAL: t{rospy.get_time()-t0:5.1f}s  dist {np.nanmean(d):5.2f}"
                  f"  mean|ex| {np.mean(exs[-40:]):.3f} mean|ey| {np.mean(eys[-40:]):.3f}"
                  f"  a=[{a[0,0]:+.2f} {a[0,1]:+.2f} {a[0,2]:+.2f} {a[0,3]:+.2f}]", flush=True)
    d = np.array(dists, dtype=float)
    print(f">>> TD3-EVAL DONE: {steps} steps, mean_dist {np.nanmean(d):.2f} "
          f"[min {np.nanmin(d):.2f} max {np.nanmax(d):.2f}], "
          f"mean|ex| {np.mean(exs):.3f} mean|ey| {np.mean(eys):.3f}", flush=True)
    rospy.signal_shutdown("eval done")


if __name__ == '__main__':
    main()
