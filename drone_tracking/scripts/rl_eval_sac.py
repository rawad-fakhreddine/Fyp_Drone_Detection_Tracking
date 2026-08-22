#!/usr/bin/env python3
"""
rl_eval_sac.py — RL Milestone (Config 3) · deterministic evaluation of a trained
SAC policy, for the motion-quality head-to-head vs IBVS.

Loads a trained SAC .zip and drives the drone through the SAME wrapper used in
training (DummyVecEnv + VecFrameStack N=4) with deterministic=True (mean action,
NO exploration), RESET-FREE, for a fixed wall/sim duration. The normal
flight_logger keeps writing ~/flight_log_latest.csv in the standard 65-col
format, so the SAME analyzer (smoothness.py) can score this run against the IBVS
baseline in the identical clean-tracking regime.

Handoff (identical to rl_test_episodes / rl_train_sac): start this so it is
publishing setpoints, then `rosnode kill /ibvs_controller_node`.

Usage:
  rosrun drone_tracking rl_eval_sac.py \
      _model:=~/fyp/rl/models/sac/sac_policy.zip _secs:=120
"""
import os
import numpy as np
import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl_env import DroneTrackingEnv


def main():
    rospy.init_node('rl_eval_sac', anonymous=True)
    model_p = os.path.expanduser(rospy.get_param('~model', '~/fyp/rl/models/sac/sac_policy.zip'))
    secs = float(rospy.get_param('~secs', 120.0))

    rospy.loginfo(f"[eval] loading SAC policy: {model_p}")
    model = SAC.load(model_p, device='cpu')

    # explicit-rate obs (n_stack=1, 2026-08-18). episode_secs huge => reset-free.
    n_stack = int(rospy.get_param('~n_stack', 1))
    env = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv(episode_secs=1e9)]), n_stack=n_stack)
    obs = env.reset()
    print(f"[eval] reset OK, obs {obs.shape} (n_stack={n_stack}); deterministic run for {secs:.0f}s",
          flush=True)

    t0 = rospy.get_time()
    steps = 0
    dists, exs, eys = [], [], []
    print("SAC-EVAL-READY", flush=True)   # handoff marker: kill IBVS now
    while not rospy.is_shutdown() and (rospy.get_time() - t0) < secs:
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(a)
        inf = info[0]
        steps += 1
        dists.append(inf.get('true_dist', float('nan')))
        exs.append(abs(float(obs[0, -16]))); eys.append(abs(float(obs[0, -15])))
        if steps % 40 == 0:
            d = np.array(dists[-40:], dtype=float)
            print(f"SAC-EVAL: t{rospy.get_time()-t0:5.1f}s  dist {np.nanmean(d):5.2f}"
                  f"  mean|ex| {np.mean(exs[-40:]):.3f} mean|ey| {np.mean(eys[-40:]):.3f}"
                  f"  a=[{a[0,0]:+.2f} {a[0,1]:+.2f} {a[0,2]:+.2f} {a[0,3]:+.2f}]", flush=True)
    d = np.array(dists, dtype=float)
    print(f">>> SAC-EVAL DONE: {steps} steps, mean_dist {np.nanmean(d):.2f} "
          f"[min {np.nanmin(d):.2f} max {np.nanmax(d):.2f}], "
          f"mean|ex| {np.mean(exs):.3f} mean|ey| {np.mean(eys):.3f}", flush=True)
    rospy.signal_shutdown("eval done")


if __name__ == '__main__':
    main()
