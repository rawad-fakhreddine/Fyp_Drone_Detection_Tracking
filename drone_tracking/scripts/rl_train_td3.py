#!/usr/bin/env python3
"""
rl_train_td3.py — RL Milestone (Config 3) · online TD3 training.

WHY TD3 (2026-08-26): pure-SAC's DETERMINISTIC mean relaxes out of the [6,7] m band
on every lever we tried this session, even though the STOCHASTIC policy holds it during
training. Root cause: SAC optimises a stochastic actor; deterministic eval takes the
MEAN, which is NOT the band-holding behaviour. TD3 optimises a DETERMINISTIC actor
directly — the policy it trains is exactly the policy it deploys, so that gap cannot
exist. Pure RL from scratch (NO behaviour cloning): the doctor rejects BC; TD3 learns
only from the reward. Exploration is external Gaussian action noise during training only.

This is ADDITIVE (CLAUDE.md hard rule): new file, ZERO edits to the IBVS/KF/mover nodes,
and it reuses rl_env.DroneTrackingEnv + the same reward + the GT replay-SEEDING helper
(fills the buffer with real approach transitions labelled by the real reward — this is
buffer seeding a la DDPGfD, NOT imitation: the actor is never supervised toward the GT
action; the reward alone drives the policy).

PRE-FLIGHT: launched by rl_td3_train_launch.sh, which brings up the sim stack (SKIP_IBVS),
holds OFFBOARD with a hover publisher, waits for takeoff, then starts this node.

TD3 knobs that matter here:
  - deterministic actor (no log_std head)  -> no stochastic-vs-deterministic gap
  - NormalActionNoise(sigma) for EXPLORATION only (training); eval is noise-free
  - twin critics + target-policy smoothing + delayed (policy_delay) actor updates
  - learning_starts=0 with GT-prefill seeding (same rationale as --scratch SAC: random
    uniform vx at cap flies the drone off-island, filling the buffer with only losses)
"""
import os, argparse
import numpy as np


def build(seed=0, tb=True, n_stack=1, train_freq=1, gradient_steps=1,
          batch_size=256, episode_secs=35.0, action_noise=0.20, policy_delay=2,
          target_policy_noise=0.20, target_noise_clip=0.5, learning_rate=3e-4,
          caps_lambda_s=0.0, caps_sigma_s=0.05, caps_lambda_t=0.0, bc_anchor=False):
    """Construct a fresh TD3 model + the same VecFrameStack(DroneTrackingEnv) as SAC.

    CAPS (caps_lambda_s>0 or caps_lambda_t>0) OR bc_anchor swaps in CAPSTD3 = TD3 + extra
    actor-loss terms (CAPS smoothness and/or annealed BC anchor). All off -> plain TD3."""
    import rospy  # noqa: F401  (node already init'd by caller)
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    _use_caps = (caps_lambda_s > 0.0) or (caps_lambda_t > 0.0) or bc_anchor
    if _use_caps:
        from rl_caps_td3 import CAPSTD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
    from rl_env import DroneTrackingEnv

    _ep = episode_secs
    venv = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv(episode_secs=_ep)]), n_stack=n_stack)
    venv = VecMonitor(venv)

    n_act = venv.action_space.shape[-1]
    # Gaussian exploration noise on the DETERMINISTIC action (training only). sigma in
    # NORMALISED action space [-1,1]; 0.20 ~ gentle exploration around the current policy
    # (a 1-sigma vx draw ~= 0.20*max_vx). Eval passes deterministic=True -> zero noise.
    noise = NormalActionNoise(mean=np.zeros(n_act, dtype=np.float32),
                              sigma=action_noise * np.ones(n_act, dtype=np.float32))

    kw = dict(learning_rate=learning_rate, buffer_size=200_000, batch_size=batch_size,
              learning_starts=0,                 # GT-prefill seeds the buffer instead
              train_freq=train_freq, gradient_steps=gradient_steps,
              tau=0.005, gamma=0.99,
              action_noise=noise,
              policy_delay=policy_delay,          # TD3: delayed actor updates
              target_policy_noise=target_policy_noise,  # TD3: target-smoothing noise
              target_noise_clip=target_noise_clip,
              policy_kwargs=dict(net_arch=[256, 256]),
              seed=seed, verbose=1)
    if tb:
        try:
            import tensorboard  # noqa: F401
            kw['tensorboard_log'] = os.path.expanduser('~/fyp/rl/logs')
        except ImportError:
            print("[td3] tensorboard not installed -> stdout logging only")

    if _use_caps:
        model = CAPSTD3("MlpPolicy", venv, **kw)
        model._caps(l_s=caps_lambda_s, s_s=caps_sigma_s, l_t=caps_lambda_t)
        print("[td3] CAPS smoothness ENABLED (spatial+temporal actor-loss regularizers)")
    else:
        model = TD3("MlpPolicy", venv, **kw)
    print(f"[td3] built: deterministic actor, twin critics, policy_delay={policy_delay}, "
          f"expl sigma={action_noise}, target_smooth={target_policy_noise}")
    return model, venv


def train(total_steps, chunk_steps, save_dir, seed, resume, n_stack=1,
          train_freq=1, gradient_steps=1, batch_size=256, episode_secs=35.0,
          gt_prefill_steps=3000, offline_pretrain=5000, action_noise=0.20, policy_delay=2,
          target_policy_noise=0.20, target_noise_clip=0.5, learning_rate=3e-4,
          caps_lambda_s=0.0, caps_sigma_s=0.05, caps_lambda_t=0.0,
          bc_anchor_path=None, bc_w0=0.0, bc_anneal=20000):
    import rospy  # noqa: F401
    from stable_baselines3 import TD3
    _bc_on = (bc_anchor_path is not None) and (bc_w0 > 0.0)
    _use_caps = (caps_lambda_s > 0.0) or (caps_lambda_t > 0.0) or _bc_on
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    # GT replay-seeding helper is algorithm-agnostic (fills the SB3 replay buffer with
    # env transitions + real rewards); reuse it rather than duplicate. NOT imitation.
    from rl_train_sac import gt_prefill_live

    save_dir = os.path.expanduser(save_dir); os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'td3_policy')
    buf_path   = os.path.join(save_dir, 'td3_replay.pkl')

    model, venv = build(seed=seed, tb=True, n_stack=n_stack, train_freq=train_freq,
                        gradient_steps=gradient_steps, batch_size=batch_size,
                        episode_secs=episode_secs, action_noise=action_noise,
                        policy_delay=policy_delay, target_policy_noise=target_policy_noise,
                        target_noise_clip=target_noise_clip, learning_rate=learning_rate,
                        caps_lambda_s=caps_lambda_s, caps_sigma_s=caps_sigma_s,
                        caps_lambda_t=caps_lambda_t, bc_anchor=_bc_on)
    reset_num = True
    if resume and os.path.exists(model_path + '.zip'):
        print(f"[td3] RESUME from {model_path}.zip")
        # CAPS: load into CAPSTD3 (keeps the train() override) and re-apply the caps weights
        # (they're plain attributes, not saved in the SB3 checkpoint). A checkpoint trained as
        # plain TD3 loads fine into CAPSTD3 — same net arch — so we can add CAPS on a resume.
        # custom_objects overrides the CHECKPOINT's saved spaces with the live env's. Needed for a
        # BC-surgered warm-start zip (saved with default Box(-inf,inf)) vs the env's Box(-3,3) —
        # same (14,) shape, only bounds differ, so this is safe. No-op for a normal resume (spaces
        # already match) — SB3 just reuses the identical space.
        _co = {"observation_space": venv.observation_space, "action_space": venv.action_space}
        if _use_caps:
            from rl_caps_td3 import CAPSTD3
            model = CAPSTD3.load(model_path, env=venv, custom_objects=_co)
            model._caps(l_s=caps_lambda_s, s_s=caps_sigma_s, l_t=caps_lambda_t)
        else:
            model = TD3.load(model_path, env=venv, custom_objects=_co)
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"[td3] replay buffer restored ({model.replay_buffer.size()} transitions)")
        # TD3.load restores the CHECKPOINT's action_noise (ignoring the CLI value) — same
        # class of bug as SAC's ent_coef-on-resume. Re-apply the requested exploration sigma
        # so a resume meant to EXPLOIT (lower noise, to settle bang-bang oscillation) really
        # lowers it. NormalActionNoise is stateless -> a fresh one is safe.
        from stable_baselines3.common.noise import NormalActionNoise
        _na = venv.action_space.shape[-1]
        model.action_noise = NormalActionNoise(mean=np.zeros(_na, dtype=np.float32),
                                               sigma=action_noise * np.ones(_na, dtype=np.float32))
        print(f"[td3] resume: re-applied exploration sigma={action_noise}")
        # Re-apply LR + gradient_steps on resume: TD3.load restores the CHECKPOINT's optimizer
        # LR (via lr_schedule) and UTD ratio, IGNORING the CLI values (same bug class as
        # action_noise above). For an anti-divergence resume we MUST actually lower them.
        # (2026-08-31: the mix(1,2,4,8) run diverged past ~180k under lr=3e-4/gradient_steps=2;
        # resume 164k at lr=1e-4/gs=1 to stabilise -> see [[rl-td3-diverges-after-180k-peak]].)
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        for _opt in (model.actor.optimizer, model.critic.optimizer):
            for _g in _opt.param_groups:
                _g['lr'] = learning_rate
        model.gradient_steps = gradient_steps
        print(f"[td3] resume: re-applied learning_rate={learning_rate} gradient_steps={gradient_steps}")
        # BC-WARM-START case: a staged td3_policy.zip (BC-surgered actor) has NO matching replay
        # buffer, so the critic is random -> its bad Q-gradients would wreck the good warm-started
        # actor on the first online updates. Seed the buffer from GT + offline-pretrain the critic
        # (DDPGfD warmup) so Q is sane before going online. Guarded on size < batch_size so a
        # NORMAL resume (buffer restored above) skips this untouched.
        if model.replay_buffer.size() < batch_size:
            print("[td3] resume buffer empty (BC warm-start) -> GT-prefill + offline pretrain to seed critic")
            gt_prefill_live(model, venv, n_steps=gt_prefill_steps)
            if offline_pretrain > 0 and model.replay_buffer.size() > batch_size:
                from stable_baselines3.common.logger import configure
                model.set_logger(configure(None, ['stdout']))
                model.train(gradient_steps=offline_pretrain, batch_size=model.batch_size)
                print(f"[td3] BC-warm-start offline pretrain done (n_updates={model._n_updates})")
        reset_num = False
    else:
        # fresh run: GT-heuristic prefill (buffer seeding). Random exploration at vx_cap
        # flies off-island; the GT approach controller gives real approach->detect->reward
        # transitions so the critic starts useful. The actor is NEVER trained on the GT
        # action — only on the reward -> pure RL.
        gt_prefill_live(model, venv, n_steps=gt_prefill_steps)
        # OFFLINE PRETRAIN (the cold-start fix — 2026-08-26): a random-init TD3 actor
        # outputs ~0 and immediately LOSES the orbiting target every episode (first run:
        # det 2%, lost every episode, ep_rew -495) -> it collects only loss data and cannot
        # learn. Run TD3 gradient steps on the GT-SEEDED buffer BEFORE going online: the
        # critic fits the real approach->detect->reward Q, and the DPG actor update pulls
        # the actor toward APPROACH via dQ/da. Pure RL / DDPGfD-style offline warmup — the
        # actor learns from the critic's Q, NEVER supervised on the GT action (no BC).
        if offline_pretrain > 0 and model.replay_buffer.size() > batch_size:
            import time as _time
            from stable_baselines3.common.logger import configure
            model.set_logger(configure(None, ['stdout']))   # train() before learn() needs a logger
            print(f"[td3] offline pretrain: {offline_pretrain} gradient steps on "
                  f"{model.replay_buffer.size()} GT-seeded transitions "
                  f"(pure RL / DPG — actor learns from critic Q, NOT imitation)...")
            t0 = _time.time()
            model.train(gradient_steps=offline_pretrain, batch_size=model.batch_size)
            print(f"[td3] offline pretrain done in {_time.time()-t0:.0f}s "
                  f"(n_updates={model._n_updates})")

    # BC ANCHOR: load the frozen BC teacher and attach it (CAPSTD3._set_bc). Applies to BOTH the
    # fresh and resume paths — the model is a CAPSTD3 whenever _bc_on. Annealed inside train().
    if _bc_on:
        import torch as _th
        from rl_train_bc import BCPolicy
        _ckp = _th.load(os.path.expanduser(bc_anchor_path), map_location=model.device)
        _od = int(_ckp.get('obs_dim', model.observation_space.shape[0]))
        _bc = BCPolicy(obs_dim=_od).to(model.device)
        _bc.load_state_dict(_ckp['state_dict']); _bc.eval()
        model._set_bc(_bc, bc_w0, bc_anneal)
        print(f"[td3] BC anchor attached: {bc_anchor_path} obs_dim={_od} w0={bc_w0} anneal={bc_anneal}")

    ckpt = CheckpointCallback(save_freq=max(chunk_steps, 2000),
                              save_path=save_dir, name_prefix='td3_ckpt',
                              save_replay_buffer=True)

    # ---- domain-metric TensorBoard logger (track/*) — identical to the SAC trainer ----
    from collections import deque as _deque
    import numpy as _np
    from rl_env import REWARD_DEFAULTS as _RD

    class TBStats(BaseCallback):
        def __init__(self, window=2000, every=500):
            super().__init__()
            self.d = _deque(maxlen=window); self.valid = _deque(maxlen=window)
            self.n_coll = 0; self.n_lost = 0; self.every = every
            self.lo = _RD['band_lo']; self.hi = _RD['band_hi']
            self.close = _RD.get('close_thresh', 5.0)

        def _on_step(self):
            for inf in self.locals.get('infos', []):
                dt = inf.get('true_dist', float('nan'))
                if dt == dt:
                    self.d.append(dt)
                self.valid.append(int(inf.get('valid', 0)))
                r = inf.get('term_reason', '')
                if r == 'collision':   self.n_coll += 1
                elif r == 'lost':      self.n_lost += 1
            if self.num_timesteps % self.every == 0 and len(self.d) > 0:
                d = _np.asarray(self.d, dtype=float)
                self.logger.record('track/detection_frac', float(_np.mean(self.valid)))
                self.logger.record('track/d_true_mean', float(_np.mean(d)))
                self.logger.record('track/d_true_min', float(_np.min(d)))
                self.logger.record('track/d_true_max', float(_np.max(d)))
                self.logger.record('track/in_band_frac',
                                   float(_np.mean((d >= self.lo) & (d <= self.hi))))
                self.logger.record('track/too_close_frac', float(_np.mean(d < self.close)))
                self.logger.record('track/collisions_cum', self.n_coll)
                self.logger.record('track/lost_cum', self.n_lost)
            return True

    try:
        model.learn(total_timesteps=total_steps, callback=[ckpt, TBStats()],
                    reset_num_timesteps=reset_num, progress_bar=False, log_interval=1)
    finally:
        model.save(model_path)
        model.save_replay_buffer(buf_path)
        venv.close()
        print(f"[td3] saved model -> {model_path}.zip  +  replay buffer -> {buf_path}")


if __name__ == '__main__':
    import rospy
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--chunk', type=int, default=5_000)
    ap.add_argument('--save',  default='~/fyp/rl/models/td3')
    ap.add_argument('--seed',  type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--n-stack', type=int, default=1)
    ap.add_argument('--train-freq', type=int, default=1)
    ap.add_argument('--gradient-steps', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--episode-secs', type=float, default=35.0)
    ap.add_argument('--gt-prefill-steps', type=int, default=3000)
    ap.add_argument('--offline-pretrain', type=int, default=5000,
                    help='offline TD3 gradient steps on the GT-seeded buffer before going online '
                         '(cold-start fix; actor learns via critic Q / DPG — pure RL, not BC)')
    ap.add_argument('--action-noise', type=float, default=0.20,
                    help='exploration Gaussian sigma in normalised action space (training only)')
    ap.add_argument('--policy-delay', type=int, default=2, help='TD3 delayed actor update ratio')
    ap.add_argument('--target-policy-noise', type=float, default=0.20, help='TD3 target-smoothing sigma')
    ap.add_argument('--target-noise-clip', type=float, default=0.5)
    ap.add_argument('--learning-rate', type=float, default=3e-4)
    ap.add_argument('--caps-lambda-s', type=float, default=0.0,
                    help='CAPS SPATIAL smoothness weight (‖π(s)−π(s+ε)‖²); 0=off. No deployment lag.')
    ap.add_argument('--caps-sigma-s', type=float, default=0.05,
                    help='CAPS obs-perturbation std ε in normalised obs space')
    ap.add_argument('--caps-lambda-t', type=float, default=0.0,
                    help='CAPS TEMPORAL smoothness weight (‖π(s)−π(s_next)‖²); 0=off')
    ap.add_argument('--bc-anchor', default=None,
                    help='path to a BCPolicy .pth teacher; adds annealed MSE(actor,bc) to the '
                         'actor loss (accelerates off a plateau, then releases). None=off')
    ap.add_argument('--bc-w0', type=float, default=0.0,
                    help='initial BC-anchor weight (0=off); decays linearly to 0 over --bc-anneal')
    ap.add_argument('--bc-anneal', type=int, default=20000,
                    help='updates over which the BC-anchor weight decays w0 -> 0')
    import sys
    argv = [a for a in sys.argv[1:] if not a.startswith('_') and ':=' not in a]
    a = ap.parse_args(argv)

    rospy.init_node('rl_train_td3', anonymous=True)
    train(a.steps, a.chunk, a.save, a.seed, a.resume, n_stack=a.n_stack,
          train_freq=a.train_freq, gradient_steps=a.gradient_steps, batch_size=a.batch_size,
          episode_secs=a.episode_secs, gt_prefill_steps=a.gt_prefill_steps,
          offline_pretrain=a.offline_pretrain,
          action_noise=a.action_noise, policy_delay=a.policy_delay,
          target_policy_noise=a.target_policy_noise, target_noise_clip=a.target_noise_clip,
          learning_rate=a.learning_rate,
          caps_lambda_s=a.caps_lambda_s, caps_sigma_s=a.caps_sigma_s,
          caps_lambda_t=a.caps_lambda_t,
          bc_anchor_path=a.bc_anchor, bc_w0=a.bc_w0, bc_anneal=a.bc_anneal)
