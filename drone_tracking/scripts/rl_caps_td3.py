#!/usr/bin/env python3
"""
rl_caps_td3.py — TD3 + CAPS smoothness regularization (Config 3, oscillation reduction).

WHY (2026-08-27): reward-shaping smoothness (w_s, the −w_s·‖Δa‖² term) did NOT reduce the
deterministic policy's ~10 Hz action chatter across seeds — it's too weak vs the centering
reward. Output filters (accel_limit slew cap / action_lpf EMA) DO cut the jerk but add LAG on
the continuous orbit -> tracking cost. CAPS (Mysore et al. 2021, "Regularizing Action Policies
for Smooth Control") attacks the chatter at its SOURCE with ZERO deployment lag: it adds two
regularizers to the ACTOR loss during training, so the smoothness is baked into the weights.

  L_actor = −Q1(s, π(s))                                 # TD3 objective
            + λ_S · ‖π(s) − π(s + ε)‖²   , ε ~ N(0, σ_S) # SPATIAL: robust to obs noise (the
                                                          #   deterministic chatter source) — NO lag
            + λ_T · ‖π(s) − π(s')‖²      , s' = next obs  # TEMPORAL: consecutive-state smoothness

The SPATIAL term is the key no-lag lever: it flattens the policy's sensitivity to small
observation perturbations (detection jitter), so tiny obs wiggles no longer produce action
dither. Unlike an output filter it does NOT delay the response — the policy stays fully
reactive to REAL state changes, only its noise-amplification is damped.

ADDITIVE: new file, subclasses SB3 TD3, overrides only train(). λ_S=λ_T=0 -> identical to TD3.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update


class CAPSTD3(TD3):
    """TD3 with CAPS spatial + temporal action-smoothness regularizers on the actor loss."""

    # set by the trainer after construction/load (attributes, so they survive .load())
    # NOTE (2026-08-28): caps_lambda_s is now an ADAPTIVE TARGET FRACTION, not a raw weight.
    # Each batch, λ_s is auto-scaled so the spatial smoothness term contributes ~caps_lambda_s
    # of the |PG-loss| magnitude (e.g. 0.10 = ~10%). This replaces the brittle raw λ=150 that
    # cut jerk on seed42 (-62%) but blew up on seed43 (+158%) — target-fraction is scale/seed
    # robust. Typical caps_lambda_s in [0.05, 0.15]. caps_lambda_max clamps the effective λ.
    caps_lambda_s: float = 0.0     # spatial TARGET FRACTION of |PG loss| (0 -> off)
    caps_sigma_s: float = 0.05     # obs-perturbation std (normalized obs space)
    caps_lambda_t: float = 0.0     # temporal weight (raw; kept off in the ws3+CAPS run)
    caps_lambda_max: float = 300.0 # clamp on the effective adaptive λ_s (guards tiny-l_s blowup)

    # --- BC ANCHOR (2026-09-04): annealed imitation pull toward a FROZEN BC teacher, added to the
    # actor loss. L += λ_bc(t)·MSE(π(s), bc(s)), λ_bc = bc_w0·max(0, 1 − n_updates/bc_anneal).
    # Accelerates learning of the teacher's fast T3 pursuit off a plateau, then RELEASES (λ→0) so
    # RL SURPASSES the teacher (C3 must beat C1, not equal it). bc_net None / bc_w0=0 -> off.
    bc_net = None                  # frozen BCPolicy (set via _set_bc); teacher outputs in [-1,1]
    bc_w0: float = 0.0             # anchor TARGET FRACTION of |PG loss| at start (0 -> off; ~0.5 typical)
    bc_anneal: int = 20000         # updates over which the anchor fraction decays w0 -> 0
    bc_start_updates: int = 0      # _n_updates when the anchor was attached (anneal is RELATIVE to
                                   # this — on a RESUME _n_updates is already large, so an absolute
                                   # anneal would be 0 from step 1; must count from attach time)

    def _set_bc(self, bc_net, w0, anneal):
        self.bc_net = bc_net
        self.bc_w0 = float(w0)
        self.bc_anneal = int(anneal)
        self.bc_start_updates = int(self._n_updates)
        if bc_net is not None:
            for p in bc_net.parameters():
                p.requires_grad_(False)
            bc_net.eval()
        print(f"[bc-anchor] w0={self.bc_w0} anneal={self.bc_anneal} net={'set' if bc_net is not None else 'None'}")
        return self

    def _caps(self, l_s=None, s_s=None, l_t=None):
        if l_s is not None: self.caps_lambda_s = float(l_s)
        if s_s is not None: self.caps_sigma_s = float(s_s)
        if l_t is not None: self.caps_lambda_t = float(l_t)
        print(f"[caps] lambda_s={self.caps_lambda_s} sigma_s={self.caps_sigma_s} "
              f"lambda_t={self.caps_lambda_t}")
        return self

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # ---- verbatim SB3 v2.4.1 TD3.train(), with CAPS added to the actor loss ----
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, caps_s_losses, caps_t_losses = [], [], [], []
        caps_lam_eff = []
        bc_losses, bc_lam_eff = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                obs = replay_data.observations
                pi = self.actor(obs)
                actor_loss = -self.critic.q1_forward(obs, pi).mean()

                # --- BC ANCHOR (annealed): pull the actor toward the frozen BC teacher, weight
                # bc_w0 -> 0 over bc_anneal updates. Early: keeps the policy near C1's fast T3
                # pursuit while the critic sharpens (accelerates off the plateau). Late: releases
                # so RL surpasses C1. Added BEFORE CAPS so CAPS's PG-magnitude scaling sees it. ---
                if self.bc_net is not None and self.bc_w0 > 0.0:
                    _elapsed = self._n_updates - self.bc_start_updates   # updates since attach
                    frac_anneal = max(0.0, 1.0 - _elapsed / float(max(1, self.bc_anneal)))
                    if frac_anneal > 0.0:
                        with th.no_grad():
                            bc_act = self.bc_net(obs)
                        l_bc = F.mse_loss(pi, bc_act)
                        # bc_w0 = TARGET FRACTION of |PG loss| (e.g. 0.5 = BC term ~50% of PG at
                        # the start), scaled down by the linear anneal -> 0. Scale/seed-robust,
                        # same trick as CAPS spatial: auto-solve λ so λ·l_bc = w0·anneal·|PG|.
                        pg_mag = actor_loss.detach().abs()
                        lam_bc = float((self.bc_w0 * frac_anneal * pg_mag
                                        / (l_bc.detach() + 1e-8)).clamp(0.0, 1e4))
                        actor_loss = actor_loss + lam_bc * l_bc
                        bc_losses.append(l_bc.item()); bc_lam_eff.append(lam_bc)

                # --- CAPS spatial (ADAPTIVE): penalize action change under a small obs
                # perturbation, with λ_s auto-scaled so this term ~= caps_lambda_s * |PG loss|
                # each batch (scale/seed-robust; see class note). ---
                if self.caps_lambda_s > 0.0:
                    obs_pert = obs + th.randn_like(obs) * self.caps_sigma_s
                    pi_pert = self.actor(obs_pert)
                    l_s = F.mse_loss(pi, pi_pert)
                    pg_mag = actor_loss.detach().abs()               # |PG loss| before CAPS
                    lam_s = float((self.caps_lambda_s * pg_mag / (l_s.detach() + 1e-8))
                                  .clamp(0.0, self.caps_lambda_max))
                    actor_loss = actor_loss + lam_s * l_s
                    caps_s_losses.append(l_s.item())
                    caps_lam_eff.append(lam_s)

                # --- CAPS temporal: penalize action change between consecutive states ---
                if self.caps_lambda_t > 0.0:
                    pi_next = self.actor(replay_data.next_observations)
                    l_t = F.mse_loss(pi, pi_next)
                    actor_loss = actor_loss + self.caps_lambda_t * l_t
                    caps_t_losses.append(l_t.item())

                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(caps_s_losses) > 0:
            self.logger.record("train/caps_spatial", np.mean(caps_s_losses))
            self.logger.record("train/caps_lambda_eff", np.mean(caps_lam_eff))
        if len(caps_t_losses) > 0:
            self.logger.record("train/caps_temporal", np.mean(caps_t_losses))
        if len(bc_losses) > 0:
            self.logger.record("train/bc_anchor_mse", np.mean(bc_losses))
            self.logger.record("train/bc_lambda_eff", np.mean(bc_lam_eff))
