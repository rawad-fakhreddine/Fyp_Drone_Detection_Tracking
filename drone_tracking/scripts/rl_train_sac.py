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
    obs_dim = int(ck.get('obs_dim', sd['net.0.weight'].shape[1]))   # 64 (old stack) or 14 (v4)
    net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 4), nn.Tanh()).to(device)
    with torch.no_grad():
        net[0].weight.copy_(sd['net.0.weight']); net[0].bias.copy_(sd['net.0.bias'])
        net[2].weight.copy_(sd['net.2.weight']); net[2].bias.copy_(sd['net.2.bias'])
        net[4].weight.copy_(sd['net.4.weight']); net[4].bias.copy_(sd['net.4.bias'])
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net


def _make_sacbc(bc_ref, bc_w0, bc_alpha, anneal, aprev_idx=None, zero_aprev_ref=True):
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

    # a_prev slots in the obs. Old 64-dim 4-frame stack: a_prev at 12:16 in each frame.
    # New 14-dim explicit-rate single frame (v4): a_prev at 10:14. Passed in from the call
    # site (derived from the BC checkpoint's obs_dim/stack_n) so this generalizes.
    # Zeroing them in the ANCHOR TARGET pulls the actor toward the NON-amplifying response
    # (BC amplifies a_prev = the runaway; BC(a_prev=0) is stable) while the ACTOR still
    # SEES real a_prev for tracking. Diagnosed 2026-08-17.
    APREV_IDX = list(aprev_idx) if aprev_idx is not None else \
                [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63]

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


def _gt_action(ob):
    """GT-heuristic action in [-1,1]: approach target at d=6.5m using world position."""
    import math as _math
    rel = ob.rel_w; yaw = ob.chaser_yaw; caps = ob.caps
    if any(_math.isnan(v) for v in rel):
        return np.zeros(4, dtype=np.float32)
    dx_w, dy_w, dz_w = rel           # ENU world: dx=East, dy=North, dz=Up
    rng_xy = _math.hypot(dx_w, dy_w)
    d_star  = 6.5
    if rng_xy > 0.1:
        ue, un = dx_w / rng_xy, dy_w / rng_xy
        # ENU→body FRD: yaw measured CCW from East
        fwd_unit   =  ue * _math.cos(yaw) + un * _math.sin(yaw)
        right_unit =  ue * _math.sin(yaw) - un * _math.cos(yaw)
        speed = max(-4.0, min(4.0, 1.5 * (rng_xy - d_star)))
        vx = speed * fwd_unit;  vy = speed * right_unit
    else:
        vx = vy = 0.0
    vz = max(-2.5, min(2.5, 0.8 * dz_w))   # ENU via MAVROS: +vz=UP; dz_w=ENU up → climb toward target
    wz = 0.0
    if rng_xy > 0.5:
        tgt_yaw = _math.atan2(dy_w, dx_w)
        dyaw    = _math.atan2(_math.sin(tgt_yaw - yaw), _math.cos(tgt_yaw - yaw))
        # PROBE-VERIFIED 2026-08-24: setpoint_raw/local yaw_rate is ENU (CCW positive).
        # dyaw>0 (target to left/CCW) needs a CCW turn = POSITIVE yaw_rate.
        wz      = max(-1.0, min(1.0, 1.5 * dyaw))
    return np.clip([vx/caps[0], vy/caps[1], vz/caps[2], wz/caps[3]],
                   -1.0, 1.0).astype(np.float32)


def gt_prefill_live(model, venv, n_steps=3000):
    """Fill replay buffer with GT-heuristic approach transitions before SAC training.

    Runs N steps using the GT position vector to command approach to 6.5 m, storing
    each transition in the SB3 replay buffer.  No SAC gradients — the policy is not
    used.  This gives the critic real 'approach→detect→reward' examples so training
    can start from a useful prior instead of the random hover the policy produces.
    """
    # unwrap VecMonitor → DummyVecEnv → DroneTrackingEnv
    inner = venv
    while hasattr(inner, 'venv'):
        inner = inner.venv
    raw_env = inner.envs[0]

    obs = venv.reset()
    added = 0
    print(f"[sac] GT-prefill: collecting {n_steps} steps with GT approach controller...")
    for _ in range(n_steps):
        action_np  = _gt_action(raw_env.ob)          # (4,) in [-1,1]
        action_b   = action_np[None].astype(np.float32)  # (1,4)
        new_obs, rewards, dones, infos = venv.step(action_b)
        model.replay_buffer.add(
            obs.astype(np.float32), new_obs.astype(np.float32),
            action_b, rewards.reshape(-1).astype(np.float32),
            dones.reshape(-1),
            [{'TimeLimit.truncated': bool(infos[0].get('TimeLimit.truncated', False))}])
        obs = new_obs
        added += 1
    print(f"[sac] GT-prefill done: {added} transitions in buffer "
          f"(size={model.replay_buffer.size()})")
    return added


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
            a = Y[i0]; prev_a = X[i0, -4:]        # a_prev = last 4 obs cols (14-dim v4: 10:14; 16-dim: 12:16)
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
          train_freq=1, gradient_steps=1, batch_size=512, episode_secs=25.0):
    import rospy, torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
    from rl_env import DroneTrackingEnv

    # n_stack=1 == EXPLICIT-RATE observation (no frame-stacking; rates carry the
    # temporal info) — the 2026-08-18 supervisor design. n_stack>1 restores stacking.
    _ep = episode_secs
    venv = VecFrameStack(DummyVecEnv([lambda: DroneTrackingEnv(episode_secs=_ep)]), n_stack=n_stack)
    venv = VecMonitor(venv)                       # logs ep_rew_mean / ep_len_mean

    # ent_coef: FIXED small (default 0.05), NOT SB3's "auto". The first live run showed
    # "auto" drives the entropy coefficient up to ~0.45 within ~2000 steps, exploring at
    # the velocity caps (±8 m/s thrash) and WASHING OUT the BC warm-start (ep_rew_mean
    # collapsed +278 -> -268). A fixed small coef keeps exploration GENTLE around the
    # cloned policy; the critic prefill (below) supplies the improvement signal instead.
    kw = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=batch_size,
              learning_starts=0,                  # GT-prefill warms buffer; no random exploration
              train_freq=train_freq, gradient_steps=gradient_steps, tau=0.005, gamma=0.99,
              ent_coef=ent_coef,
              policy_kwargs=dict(net_arch=[256, 256]),  # matches BCPolicy dims
              seed=seed, verbose=1)
    # TARGET_ENTROPY override (stability lever, 2026-08-24): only active with ent_coef="auto".
    # SB3's default target_entropy=-action_dim=-4 keeps auto ent_coef ~0.1, which lets a
    # CONVERGED policy keep exploring and WANDER off its good state late in training (the
    # recurring detection-collapse ~10-13k seen in s2b/s3/s4/s5). A lower (more negative)
    # target makes auto ent_coef decay further → policy exploits sooner → less late wander,
    # while still exploring early (auto starts high). Set via env var TARGET_ENTROPY (e.g. -8).
    _tent = os.environ.get('TARGET_ENTROPY', '').strip()
    if _tent:
        kw['target_entropy'] = float(_tent)
        print(f"[sac] target_entropy override = {_tent} (lower = exploit sooner, less late wander)")
    if tb:                                         # tensorboard is optional (may be absent)
        try:
            import tensorboard  # noqa: F401
            kw['tensorboard_log'] = os.path.expanduser('~/fyp/rl/logs')
        except ImportError:
            print("[sac] tensorboard not installed -> logging to stdout only")

    if anchor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # derive a_prev indices from the BC checkpoint layout (14-dim single-frame v4 -> [10,11,12,13];
        # 64-dim 4-frame -> [12..15,28..31,44..47,60..63]) so the anchor generalizes.
        _ck = torch.load(os.path.expanduser(anchor['ref']), map_location='cpu', weights_only=False)
        _sn = int(_ck.get('stack_n', 4)); _od = int(_ck.get('obs_dim', 64)); _fr = _od // max(_sn, 1)
        _apidx = [k*_fr + _fr - 4 + j for k in range(_sn) for j in range(4)]
        bc_ref = _load_bc_ref(anchor['ref'], device)
        SACBC = _make_sacbc(bc_ref, anchor['w0'], anchor['alpha'], anchor['anneal'], aprev_idx=_apidx)
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
          train_freq=1, gradient_steps=1, batch_size=512, episode_secs=25.0,
          gt_prefill_steps=3000):
    import rospy
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    save_dir = os.path.expanduser(save_dir); os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'sac_policy')
    buf_path   = os.path.join(save_dir, 'sac_replay.pkl')

    model, venv = build(bc_path, seed=seed, tb=True, ent_coef=ent_coef, anchor=anchor,
                        log_std=log_std, n_stack=n_stack, train_freq=train_freq,
                        gradient_steps=gradient_steps, batch_size=batch_size,
                        episode_secs=episode_secs)
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
        # FORCE fixed ent_coef on resume when a numeric --ent-coef is given. SAC.load restores
        # the checkpoint's entropy config (often "auto"), silently ignoring the CLI value — so a
        # resume meant to EXPLOIT (low fixed ent_coef) would keep auto-exploring at the old high
        # coef (observed: --ent-coef 0.02 resume ran at ent_coef≈0.14). Override the loaded model.
        if not (isinstance(ent_coef, str) and str(ent_coef).startswith('auto')):
            import torch as th
            # ent_coef (the hyperparam string/float) must ALSO become the float, or SAC.load on
            # the resulting checkpoint rebuilds an AUTO model that expects an ent_coef_optimizer the
            # saved torch params no longer have → load KeyError. Setting it float makes reload build
            # a fixed-ent model that matches. (Recover an already-broken zip via SAC.load with
            # custom_objects={'ent_coef': <float>}.)
            model.ent_coef = float(ent_coef)
            model.ent_coef_optimizer = None
            model.log_ent_coef = None
            model.ent_coef_tensor = th.tensor(float(ent_coef), device=model.device)
            print(f"[sac] resume: FORCED fixed ent_coef = {float(ent_coef):.4f} (exploit, was auto)")
        reset_num = False
    elif not resume and not prefill:
        # SCRATCH mode: GT-heuristic prefill — approach target using GT position to
        # seed the buffer with 'approach→detect→reward' transitions before SAC training.
        # Random-action exploration (learning_starts>0) fails because vx_cap=8m/s flies
        # the drone 40m away in 5s, filling the buffer with only loss-timeout episodes.
        gt_prefill_live(model, venv, n_steps=gt_prefill_steps)
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
            # CRITICAL (2026-09-04): the stdout logger above set _custom_logger=True, so
            # learn() would KEEP it and NEVER create the TensorBoard run dir (supervisor
            # requires TB live). Clear the flag so _setup_learn rebuilds the proper
            # TB+stdout logger from self.tensorboard_log.
            model._custom_logger = False

    ckpt = CheckpointCallback(save_freq=max(chunk_steps, 2000),
                              save_path=save_dir, name_prefix='sac_ckpt',
                              save_replay_buffer=True)

    # ---- domain-metric TensorBoard logger (track/*) ----
    # SB3 already logs rollout/ep_rew_mean, ep_len_mean, train/* losses. This adds the
    # tracking metrics we care about (detection %, d_true stats, in-band/too-close %,
    # collision/lost counts) so ALL progress is visible as curves, not just grepped.
    from collections import deque as _deque
    import numpy as _np
    from rl_env import REWARD_DEFAULTS as _RD

    class TBStats(BaseCallback):
        def __init__(self, window=2000, every=500):
            super().__init__()
            self.d = _deque(maxlen=window); self.valid = _deque(maxlen=window)
            # centering + speed windows (supervisor 2026-09-04: watch ex/ey off-center and
            # whether the chaser keeps up with the target). Only accumulate on VALID frames
            # (ex/ey/d meaningless without a detection; speeds are GT so always valid).
            self.ex = _deque(maxlen=window); self.ey = _deque(maxlen=window)
            self.cspd = _deque(maxlen=window); self.tspd = _deque(maxlen=window)
            self.n_coll = 0; self.n_lost = 0; self.every = every
            self.lo = _RD['band_lo']; self.hi = _RD['band_hi']
            self.close = _RD.get('close_thresh', 5.0)

        def _on_step(self):
            for inf in self.locals.get('infos', []):
                dt = inf.get('true_dist', float('nan'))
                if dt == dt:                       # not NaN
                    self.d.append(dt)
                v = int(inf.get('valid', 0))
                self.valid.append(v)
                if v:                              # centering only meaningful when detecting
                    ex = inf.get('ex', float('nan')); ey = inf.get('ey_c', float('nan'))
                    if ex == ex: self.ex.append(abs(ex))
                    if ey == ey: self.ey.append(abs(ey))
                cs = inf.get('chaser_spd', float('nan')); ts = inf.get('target_spd', float('nan'))
                if cs == cs: self.cspd.append(cs)
                if ts == ts: self.tspd.append(ts)
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
                # --- centering (ex/ey) ---
                if len(self.ex) > 0:
                    ax = _np.asarray(self.ex); ay = _np.asarray(self.ey)
                    self.logger.record('track/ex_abs_mean', float(_np.mean(ax)))
                    self.logger.record('track/ey_abs_mean', float(_np.mean(ay)))
                    # visual-lock = centered (|ex|<0.30 AND |ey_c|<0.30), the HOLD centering gate
                    n = min(len(ax), len(ay))
                    self.logger.record('track/centered_frac',
                                       float(_np.mean((ax[:n] < 0.30) & (ay[:n] < 0.30))))
                # --- chaser vs target speed (does the chaser keep up?) ---
                if len(self.cspd) > 0 and len(self.tspd) > 0:
                    cm = float(_np.mean(self.cspd)); tm = float(_np.mean(self.tspd))
                    self.logger.record('track/chaser_spd_mean', cm)
                    self.logger.record('track/target_spd_mean', tm)
                    self.logger.record('track/speed_deficit', tm - cm)  # >0 = falling behind
            return True

    # ---- STOP-ON-MAX-REWARD (supervisor directive 2026-09-04) ----
    # Do NOT stop the run because a step/time budget expired — keep training and stop
    # when the reward has MAXED OUT (plateaued). Every check window we read the Monitor's
    # rolling ep_rew_mean; a new max saves sac_best.zip (+buffer); no improvement for
    # `patience` consecutive checks => the reward has converged at its max => stop.
    # An optional hard REWARD_THRESHOLD stops immediately once reached. All env-tunable.
    class StopOnMaxReward(BaseCallback):
        def __init__(self):
            super().__init__()
            self.check_every = int(os.environ.get('STOP_CHECK_EVERY', 2000))
            self.patience    = int(os.environ.get('STOP_PATIENCE', 25))
            self.min_delta   = float(os.environ.get('STOP_MIN_DELTA', 1.0))
            self.warmup      = int(os.environ.get('STOP_WARMUP', 15000))
            _thr = os.environ.get('REWARD_THRESHOLD', '').strip()
            self.threshold   = float(_thr) if _thr else None
            self.best = -1e18; self.no_improve = 0
            self.best_path = os.path.join(save_dir, 'sac_best')

        def _save_best(self, tag):
            self.model.save(self.best_path)
            self.model.save_replay_buffer(self.best_path + '_replay.pkl')
            print(f"[sac] *** NEW BEST ep_rew_mean={self.best:.2f} @ {self.num_timesteps} "
                  f"-> saved {self.best_path}.zip ({tag})")

        def _on_step(self):
            if self.num_timesteps % self.check_every != 0:
                return True
            buf = self.model.ep_info_buffer
            if not buf or len(buf) < 3:
                return True
            mean_r = float(np.mean([e['r'] for e in buf]))
            self.logger.record('track/ep_rew_mean_win', mean_r)
            self.logger.record('track/best_ep_rew', self.best if self.best > -1e17 else mean_r)
            self.logger.record('track/no_improve_checks', self.no_improve)
            if mean_r > self.best + self.min_delta:
                self.best = mean_r; self.no_improve = 0
                self._save_best('new-max')
            else:
                self.no_improve += 1
            if self.threshold is not None and mean_r >= self.threshold:
                print(f"[sac] STOP: ep_rew_mean {mean_r:.2f} >= REWARD_THRESHOLD "
                      f"{self.threshold:.2f} -> reward target reached")
                return False
            if self.num_timesteps >= self.warmup and self.no_improve >= self.patience:
                print(f"[sac] STOP: reward MAXED OUT — no improvement over {self.min_delta} "
                      f"for {self.patience} checks (best={self.best:.2f}, last={mean_r:.2f})")
                return False
            return True

    try:
        model.learn(total_timesteps=total_steps,
                    callback=[ckpt, TBStats(), StopOnMaxReward()],
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
    ap.add_argument('--episode-secs', type=float, default=25.0,
                    help='max seconds per episode (truncation). 25=one T4 orbit, 40=1.5 orbits')
    ap.add_argument('--gt-prefill-steps', type=int, default=3000,
                    help='GT-heuristic prefill steps for --scratch (default 3000 ~150s; use a small value for a quick smoke test)')
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
    if a.resume:
        # RESUME loads the saved SAC checkpoint (weights + replay buffer), so a BC
        # warm-start/anchor is both pointless (immediately overwritten by the load) and
        # BROKEN: the default bc_policy_v3 is 64-dim (old frame-stack) while the current
        # explicit-rate env is 14-dim (n_stack=1) → weight-copy crash. Force pure resume.
        bc_path = None; anchor = None; prefill = False
        print("[sac] RESUME: pure SAC continue from checkpoint (no warm-start/anchor/prefill)")
    rospy.init_node('rl_train_sac', anonymous=True)
    train(bc_path, a.steps, a.chunk, a.save, a.seed, a.resume,
          ent_coef=ec, prefill=prefill, critic_pretrain=a.critic_pretrain,
          anchor=anchor, log_std=a.log_std, n_stack=a.n_stack, train_freq=a.train_freq,
          gradient_steps=a.gradient_steps, batch_size=a.batch_size,
          episode_secs=a.episode_secs, gt_prefill_steps=a.gt_prefill_steps)
