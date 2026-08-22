#!/usr/bin/env python3
"""
rl_train_sac.py — RL Milestone (Config 3) · online SAC training.

The RL policy replaces the IBVS CONTROL block (YOLO perception stays). This script
runs the online SAC phase against rl_env.DroneTrackingEnv, warm-started from the
behaviour-cloning policy (bc_policy_v2.pth). Design is locked in
FYP/RL/training/Training_Design.docx; this file is the fill-in, not a redesign.

PRE-FLIGHT (must be running before this node starts):
  the full sim stack (Gazebo + PX4 + YOLO + Kalman + target_mover) with IBVS having
  taken off and reached a hold, THEN hand control to RL:
      rosrun drone_tracking rl_train_sac.py _bc:=~/fyp/rl/models/bc_policy_v2.pth &
      rosnode kill /ibvs_controller_node      # RL is already publishing -> no OFFBOARD gap

Key review-driven settings (see the Word doc "RL_Step5_SAC_Training.docx"):
  - learning_starts=0  -> the WARM-STARTED actor flies from step 1 (SB3's default 100
    would fly 100 steps of UNIFORM-RANDOM vx∈[-8,8] at the handoff, wasting the warm
    start and being unsafe). The env's OFFBOARD keepalive covers the gradient/save gaps.
  - actor log_std head warm-started to sigma≈0.14 (bias≈-2) so exploration is GENTLE
    around the BC behaviour instead of the random-init sigma≈1 that would drown it.
  - VecFrameStack(N=4): 16-dim single frame -> 64-dim policy input (oldest-first,
    matching rl_bc_dataset.stack() and the BC net's input layout).
  - chunked training: model + replay buffer are checkpointed so a run can be stopped
    and resumed (--resume) — WSL degrades over long sim sessions, and the bounded
    trajectories keep the target on the island across a chunk.

Curriculum (set the target trajectory via the launch, NOT here): start on the BOUNDED
trajectories (T1 static, T4 circle, T8 helix) which stay on the island for hours;
introduce T2/T3 (one-way, unbounded) only in short chunks, and T5/T7 last.
"""
import os, argparse, math
import numpy as np


def _warm_start(model, bc_path, log_std_init=-3.0):
    """Load the BC MLP body into the SAC actor and tame the initial exploration std.
    log_std_init sets the exploration sigma (bias): -3.0 -> sigma~0.05, -3.5 -> ~0.03,
    -4.0 -> ~0.018. Lower = gentler exploration = fewer a_prev-amplified excursions."""
    import torch
    bc_path = os.path.expanduser(bc_path)
    ck = torch.load(bc_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    actor = model.policy.actor
    with torch.no_grad():
        # BC body (64->256->256->4, dims identical to the SAC actor) -> actor
        actor.latent_pi[0].weight.copy_(sd['net.0.weight']); actor.latent_pi[0].bias.copy_(sd['net.0.bias'])
        actor.latent_pi[2].weight.copy_(sd['net.2.weight']); actor.latent_pi[2].bias.copy_(sd['net.2.bias'])
        actor.mu.weight.copy_(sd['net.4.weight']);           actor.mu.bias.copy_(sd['net.4.bias'])
        # exploration head is NOT in the BC net -> its random init gives sigma≈1, which
        # would swamp the cloned behaviour. Set it very LOW (sigma≈0.05). The diagnostic
        # run showed sigma≈0.14 is already too much: the observation carries a_prev (the
        # last action), the BC policy is heavily a_prev-dependent, and during training
        # a_prev is the NOISY exploratory action -> the noise gets amplified into runaway
        # commands (a single 2σ vx draw at σ=0.14 is ~2 m/s -> drove into the target while
        # it sat at the frame edge). Gentle exploration keeps the policy near the stable
        # clone; SAC widens it as the (prefilled) critic justifies it.
        actor.log_std.weight.mul_(0.01)
        actor.log_std.bias.fill_(log_std_init)
    print(f"[sac] warm-started actor body + log_std<-{log_std_init} (sigma~{math.exp(log_std_init):.3f}) "
          f"from {bc_path} (BC val MSE {ck.get('val_mse')})")


def _load_bc_ref(bc_path, device):
    """Frozen bc_v2 reference net (64->256->256->4, tanh) for the BC anchor. Identical
    architecture to BCPolicy / the SAC actor body, so a_bc = bc_ref(stacked_obs)."""
    import torch, torch.nn as nn
    ck = torch.load(os.path.expanduser(bc_path), map_location=device, weights_only=False)
    sd = ck['state_dict']
    net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 4), nn.Tanh()).to(device)
    with torch.no_grad():
        net[0].weight.copy_(sd['net.0.weight']); net[0].bias.copy_(sd['net.0.bias'])
        net[2].weight.copy_(sd['net.2.weight']); net[2].bias.copy_(sd['net.2.bias'])
        net[4].weight.copy_(sd['net.4.weight']); net[4].bias.copy_(sd['net.4.bias'])
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net


def _make_sacbc(bc_ref, bc_w0, bc_alpha, anneal, zero_aprev_ref=True):
    """SAC subclass with a RELAXABLE behaviour-cloning anchor on the actor loss (the fix
    for the a_prev runaway that keeps the GOOD bc_v2 tracker — see 2026-08-17). The only
    change vs SB3 SAC.train() is the actor objective:

        actor_loss = ent_coef*logp  −  lam*Q  +  bc_w * MSE(a_pi, a_bc)
        lam  = bc_alpha / |Q|.mean()        (TD3+BC Q-normalisation: keeps the Q gradient
                                             a bounded scale so the anchor reliably wins
                                             while the critic is still unreliable)
        bc_w = bc_w0 * max(0, 1 − n_updates/anneal)   (leash HIGH -> 0: training wheels)

    a_bc = frozen bc_v2 output for the SAME states -> the actor cannot diverge far from the
    known-good clone early (no exploration runaway), then is freed to EXCEED it as bc_w->0."""
    import torch as th
    import torch.nn.functional as F
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.utils import polyak_update

    # a_prev slots in the 64-dim frame-stacked obs (4 frames x 16, a_prev at 12:16 each).
    # Zeroing them in the ANCHOR TARGET pulls the actor toward bc_v2's NON-amplifying
    # response (bc_v2 amplifies a_prev ~8x = the runaway; bc_v2(a_prev=0) is stable) while
    # the ACTOR still SEES real a_prev for tracking. Diagnosed 2026-08-17.
    APREV_IDX = [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63]

    class SACBC(SAC):
        def train(self, gradient_steps, batch_size=64):
            self.policy.set_training_mode(True)
            optimizers = [self.actor.optimizer, self.critic.optimizer]
            if self.ent_coef_optimizer is not None:
                optimizers += [self.ent_coef_optimizer]
            self._update_learning_rate(optimizers)
            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses, bc_losses = [], [], []
            bc_w = bc_w0
            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                if self.use_sde:
                    self.actor.reset_noise()
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.item())
                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad(); ent_coef_loss.backward(); self.ent_coef_optimizer.step()
                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * sum(F.mse_loss(cq, target_q_values) for cq in current_q_values)
                critic_losses.append(critic_loss.item())
                self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                # --- relaxable BC anchor (the only change vs stock SAC) ---
                with th.no_grad():
                    ref_in = replay_data.observations
                    if zero_aprev_ref:                       # anchor toward the NON-amplifying response
                        ref_in = ref_in.clone(); ref_in[:, APREV_IDX] = 0.0
                    a_bc = bc_ref(ref_in)
                bc_loss = F.mse_loss(actions_pi, a_bc)
                lam = bc_alpha / (min_qf_pi.abs().mean().detach() + 1e-6)
                bc_w = bc_w0 * max(0.0, 1.0 - (self._n_updates + gradient_step) / float(anneal))
                actor_loss = (ent_coef * log_prob).mean() - (lam * min_qf_pi).mean() + bc_w * bc_loss
                actor_losses.append(actor_loss.item()); bc_losses.append(bc_loss.item())
                self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_w", bc_w)
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
    return SACBC


def prefill_from_logs(model, src='~/fyp/Results/Config1', since='2026-08-06',
                      until='2026-08-16', cap=120_000):
    """Seed the replay buffer with IBVS transitions so the CRITIC learns real Q-values
    before it can corrupt the warm-started actor (the "BC forgetting" the first live run
    exhibited: a random critic + high entropy pulled the actor off the cloned policy).
    Rewards are recomputed with compute_reward from the logs' GT distance — identical to
    the online reward. Segment ends are marked done (no bootstrap across a time gap).

    Date window [since, until): 'until' EXCLUDES 2026-08-16+ because the RL training runs
    from that date write their post-handoff exploration THRASH into Config1 with the phase
    frozen at HOLD (IBVS killed) — those rows would poison the clean-teacher prefill."""
    import glob, re, os as _os
    import numpy as np
    from rl_bc_dataset import convert_file, stack, OBS_NAMES, STACK_N
    from rl_env import compute_reward, REWARD_DEFAULTS
    rp = {k: float(v) for k, v in REWARD_DEFAULTS.items()}
    src = _os.path.expanduser(src)
    files = []
    for p in sorted(glob.glob(_os.path.join(src, 'traj*_zone*_*.csv'))):
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_', p)
        if m and since <= m.group(1) < until:
            files.append(p)
    if not files:
        print(f"[sac] prefill: NO logs under {src} since {since} — skipping (critic starts cold)")
        return 0
    added = 0
    for p in files:
        X, Y, T, D = convert_file(p)
        if len(X) < STACK_N + 1:
            continue
        Xs, idx = stack(X, T)                     # Xs (m,64) newest-frame index = idx
        for j in range(len(Xs) - 1):
            i0, i1 = int(idx[j]), int(idx[j + 1])
            contig = (i1 == i0 + 1)
            a = Y[i0]; prev_a = X[i0, 12:16]
            reward, term, _ = compute_reward(rp, a, prev_a, float(X[i0, 0]),
                                             float(X[i0, 1]), float(D[i0]), 1, 0.0)
            done = (not contig) or term           # truncate at time gaps / terminals
            # gap-cuts are artificial truncations, not real endings -> flag them so the
            # buffer's timeout handling BOOTSTRAPS the value instead of taking Q=0
            info = [{'TimeLimit.truncated': bool(done and not term)}]
            model.replay_buffer.add(
                Xs[j][None].astype(np.float32), Xs[j + 1][None].astype(np.float32),
                a[None].astype(np.float32), np.array([reward], dtype=np.float32),
                np.array([done]), info)
            added += 1
            if added >= cap:
                print(f"[sac] prefill: cap {cap} reached ({len(files)} logs scanned)")
                return added
    print(f"[sac] prefilled replay buffer with {added} IBVS transitions from {len(files)} logs")
    return added


def build(bc_path=None, seed=0, tb=True, ent_coef=0.02, anchor=None, log_std=-3.0, n_stack=1,
          train_freq=1, gradient_steps=1, batch_size=512):
    import rospy, torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
    from rl_env import DroneTrackingEnv

    # n_stack=1 == EXPLICIT-RATE observation (no frame-stacking; rates carry the
    # temporal info) — the 2026-08-18 supervisor design. n_stack>1 restores stacking.
    venv = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv()]), n_stack=n_stack)
    venv = VecMonitor(venv)                       # logs ep_rew_mean / ep_len_mean

    # ent_coef: FIXED small (default 0.05), NOT SB3's "auto". The first live run showed
    # "auto" drives the entropy coefficient up to ~0.45 within ~2000 steps, exploring at
    # the velocity caps (±8 m/s thrash) and WASHING OUT the BC warm-start (ep_rew_mean
    # collapsed +278 -> -268). A fixed small coef keeps exploration GENTLE around the
    # cloned policy; the critic prefill (below) supplies the improvement signal instead.
    kw = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=batch_size,
              learning_starts=0,                  # warm actor flies from step 1 (see header)
              train_freq=train_freq, gradient_steps=gradient_steps, tau=0.005, gamma=0.99,
              ent_coef=ent_coef,
              policy_kwargs=dict(net_arch=[256, 256]),  # matches BCPolicy dims
              seed=seed, verbose=1)
    if tb:                                         # tensorboard is optional (may be absent)
        try:
            import tensorboard  # noqa: F401
            kw['tensorboard_log'] = os.path.expanduser('~/fyp/rl/logs')
        except ImportError:
            print("[sac] tensorboard not installed -> logging to stdout only")

    if anchor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bc_ref = _load_bc_ref(anchor['ref'], device)
        SACBC = _make_sacbc(bc_ref, anchor['w0'], anchor['alpha'], anchor['anneal'])
        model = SACBC("MlpPolicy", venv, **kw)
        print(f"[sac] BC ANCHOR on: ref={anchor['ref']} w0={anchor['w0']} "
              f"alpha={anchor['alpha']} anneal={anchor['anneal']}")
    else:
        model = SAC("MlpPolicy", venv, **kw)
    if bc_path and os.path.exists(os.path.expanduser(bc_path)):
        _warm_start(model, bc_path, log_std_init=log_std)
    else:
        print(f"[sac] WARNING: no BC checkpoint at {bc_path} — training from scratch")
    return model, venv


def train(bc_path, total_steps, chunk_steps, save_dir, seed, resume, ent_coef=0.02,
          prefill=True, critic_pretrain=10_000, anchor=None, log_std=-3.0, n_stack=1,
          train_freq=1, gradient_steps=1, batch_size=512):
    import rospy
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    save_dir = os.path.expanduser(save_dir); os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'sac_policy')
    buf_path   = os.path.join(save_dir, 'sac_replay.pkl')

    model, venv = build(bc_path, seed=seed, tb=True, ent_coef=ent_coef, anchor=anchor,
                        log_std=log_std, n_stack=n_stack, train_freq=train_freq,
                        gradient_steps=gradient_steps, batch_size=batch_size)
    reset_num = True
    if resume and os.path.exists(model_path + '.zip'):
        # resume: reload weights + replay buffer onto the freshly-built env
        print(f"[sac] RESUME from {model_path}.zip")
        if anchor is not None:
            # keep the SACBC wrapper (train() override + BC leash): load the saved
            # weights INTO it rather than replacing with a plain SAC. The leash re-arms
            # and re-anneals for the new trajectory's exploration — the a_prev-amplify
            # runaway channel stays guarded while the policy adapts to a moving target.
            _loaded = SAC.load(model_path, env=venv)
            model.set_parameters(_loaded.get_parameters())
            del _loaded
            print("[sac] resumed weights into anchored SACBC (leash re-armed)")
        else:
            model = SAC.load(model_path, env=venv)
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"[sac] replay buffer restored ({model.replay_buffer.size()} transitions)")
        reset_num = False
    elif prefill:
        # fresh run: warm the critic on IBVS transitions BEFORE any online step.
        # CRITICAL (live-run lesson): filling the buffer is NOT enough — without the
        # offline gradient phase below, the critic is still RANDOM at handoff and its
        # dQ/da pushes the warm-started actor coherently wrong from the first airborne
        # step; the a_prev echo then amplifies that push to the velocity cap (~2 s
        # runaway). Run 1 (no prefill, on-policy buffer) was stable for 2000 steps;
        # runs 5-7 (prefill, no pretrain) all ran away — the missing piece was this:
        added = prefill_from_logs(model)
        if added and critic_pretrain > 0:
            # actor FROZEN -> SAC's Bellman targets use the fixed BC policy for next
            # actions = fitted Q-evaluation of the teacher-like policy. The actor
            # optimizer no-ops (grads stay None on frozen params); target nets polyak-
            # track the critic as usual. ~10k steps x batch 256 ~= 1 min on the 4060.
            import time as _time
            from stable_baselines3.common.logger import configure
            model.set_logger(configure(None, ['stdout']))   # train() before learn() needs a logger
            print(f"[sac] critic pretrain: {critic_pretrain} offline gradient steps "
                  f"(actor FROZEN) on {added} teacher transitions...")
            t0 = _time.time()
            for p in model.policy.actor.parameters():
                p.requires_grad_(False)
            model.train(gradient_steps=critic_pretrain, batch_size=model.batch_size)
            for p in model.policy.actor.parameters():
                p.requires_grad_(True)
            print(f"[sac] critic pretrain done in {_time.time()-t0:.0f}s "
                  f"(n_updates={model._n_updates})")

    ckpt = CheckpointCallback(save_freq=max(chunk_steps, 2000),
                              save_path=save_dir, name_prefix='sac_ckpt',
                              save_replay_buffer=True)
    try:
        model.learn(total_timesteps=total_steps, callback=ckpt,
                    reset_num_timesteps=reset_num, progress_bar=False,
                    log_interval=1)   # dump TensorBoard scalars EVERY episode (fast feedback)
    finally:
        # always persist on exit (Ctrl-C, watchdog abort, or completion)
        model.save(model_path)
        model.save_replay_buffer(buf_path)
        venv.close()
        print(f"[sac] saved model -> {model_path}.zip  +  replay buffer -> {buf_path}")


if __name__ == '__main__':
    import rospy
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc',    default='~/fyp/rl/models/bc_policy_v3.pth')  # v3 = a_prev-dropout BC
    ap.add_argument('--steps', type=int, default=30_000, help='total env steps this session')
    ap.add_argument('--chunk', type=int, default=10_000, help='checkpoint every N steps')
    ap.add_argument('--save',  default='~/fyp/rl/models/sac')
    ap.add_argument('--seed',  type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ent-coef', default='0.02',
                    help="fixed entropy coef (default 0.02) or 'auto' to restore SB3 auto-tuning")
    ap.add_argument('--no-prefill', action='store_true',
                    help='skip the IBVS critic warm-up (fresh runs prefill by default)')
    ap.add_argument('--critic-pretrain', type=int, default=10_000,
                    help='offline critic gradient steps (actor frozen) after the prefill')
    # --- relaxable BC anchor (TD3+BC): keep bc_v2's tracking, stop the a_prev runaway ---
    ap.add_argument('--anchor', action='store_true',
                    help='enable the relaxable BC anchor on the actor loss (recommended for v2)')
    ap.add_argument('--anchor-ref', default='~/fyp/rl/models/bc_policy_v2.pth',
                    help='frozen clone the actor is leashed toward (the GOOD tracker)')
    ap.add_argument('--bc-w0', type=float, default=15.0, help='initial anchor weight (leash strength)')
    ap.add_argument('--bc-alpha', type=float, default=2.5, help='TD3+BC Q-normalisation scale')
    ap.add_argument('--bc-anneal', type=int, default=15_000, help='updates to relax the leash to 0')
    ap.add_argument('--log-std', type=float, default=-3.0,
                    help='initial exploration log-std (-3.0 sigma~0.05, -3.5 ~0.03, -4.0 ~0.018)')
    # PURE ONLINE SAC (supervisor 2026-08-18): fresh SAC, no BC warm-start, no anchor,
    # no offline prefill — "the train now should be only SAC". Explicit-rate obs (n_stack 1).
    ap.add_argument('--scratch', action='store_true',
                    help='pure online SAC from scratch: ignore --bc/--anchor, no prefill')
    ap.add_argument('--n-stack', type=int, default=1,
                    help='frame-stack depth; 1 = explicit-rate obs (default), >1 restores stacking')
    ap.add_argument('--train-freq', type=int, default=1,
                    help='gradient every N env-steps (default 1). N=4 collects transitions faster')
    ap.add_argument('--gradient-steps', type=int, default=1,
                    help='gradient updates per train_freq cycle (default 1). >1 uses GPU more per env-step; SAC supports high update-to-data ratios')
    ap.add_argument('--batch-size', type=int, default=512,
                    help='SAC replay buffer batch size per gradient step (default 512, up from 256 for better GPU utilization)')
    # rosrun passes _params:=; strip them so argparse doesn't choke
    import sys
    argv = [a for a in sys.argv[1:] if not a.startswith('_') and ':=' not in a]
    a = ap.parse_args(argv)

    ec = a.ent_coef if str(a.ent_coef).startswith('auto') else float(a.ent_coef)
    if a.scratch:
        bc_path = None; prefill = False; anchor = None
        print("[sac] PURE ONLINE SAC (--scratch): no warm-start, no anchor, no prefill")
    else:
        bc_path = os.path.expanduser(a.bc); prefill = not a.no_prefill
        anchor = None
        if a.anchor:
            anchor = dict(ref=os.path.expanduser(a.anchor_ref), w0=a.bc_w0,
                          alpha=a.bc_alpha, anneal=a.bc_anneal)
    rospy.init_node('rl_train_sac', anonymous=True)
    train(bc_path, a.steps, a.chunk, a.save, a.seed, a.resume,
          ent_coef=ec, prefill=prefill, critic_pretrain=a.critic_pretrain,
          anchor=anchor, log_std=a.log_std, n_stack=a.n_stack, train_freq=a.train_freq,
          gradient_steps=a.gradient_steps, batch_size=a.batch_size)
