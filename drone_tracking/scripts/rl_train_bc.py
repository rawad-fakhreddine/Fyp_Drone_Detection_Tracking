#!/usr/bin/env python3
"""
rl_train_bc.py — RL Milestone (Config 3) · behaviour-cloning warm-start trainer.

Trains an MLP policy pi(stacked_obs 64) -> action 4 (tanh) to imitate the IBVS
teacher, from the dataset built by rl_bc_dataset.py. OFFLINE — no ROS needed.
The result is the WARM-START for online SAC (rl_train_sac.py), not the final
policy: RL fine-tuning then optimizes the reward past IBVS.

Architecture matches SB3 SAC's default actor body (MLP 256x256) so the weights
can be loaded into the SAC actor for the online phase.

Usage:
  python3 rl_train_bc.py --data ~/fyp/rl/datasets/bc_v1.npz \
                         --out  ~/fyp/rl/models/bc_policy_v1.pth
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class BCPolicy(nn.Module):
    """MLP 64 -> 256 -> 256 -> 4 with tanh head (same dims as SB3 SAC actor body)."""
    def __init__(self, obs_dim=64, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh())
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/fyp/rl/datasets/bc_v1.npz'))
    ap.add_argument('--out',  default=os.path.expanduser('~/fyp/rl/models/bc_policy_v1.pth'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    # anti-copycat: the teacher's command barely changes frame-to-frame at 20 Hz, so the
    # label is ~0.99-correlated with a_prev and a plain fit learns a_t ~ a_(t-1)+eps —
    # which in closed loop is a near-unity echo gain (marginal stability) that online RL
    # perturbations ride to the velocity cap (the live-run runaway). Randomly ZEROING
    # a_prev (dropout) forces the net to produce the action from the tracking features;
    # NOISING it on the remaining samples makes echoing unreliable everywhere (zeros-only
    # would let it learn "if a_prev==0 use features, else echo").
    ap.add_argument('--aprev-dropout', type=float, default=0.0,
                    help='prob of zeroing the a_prev dims per training sample')
    ap.add_argument('--aprev-noise', type=float, default=0.0,
                    help='gaussian noise std on a_prev dims of non-dropped samples')
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    Xtr, Ytr = d['train_Xs'].astype(np.float32), d['train_Ys'].astype(np.float32)
    Xva, Yva = d['val_Xs'].astype(np.float32),   d['val_Ys'].astype(np.float32)
    print(f"train {Xtr.shape}  val {Xva.shape}")

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BCPolicy(obs_dim=Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=a.batch, shuffle=True, drop_last=True)
    xva = torch.from_numpy(Xva).to(dev); yva = torch.from_numpy(Yva).to(dev)

    # a_prev dims = last 4 of each 16-dim frame in the oldest-first stack
    stack_n = int(d['stack_n']); frame = Xtr.shape[1] // stack_n
    apmask = torch.zeros(Xtr.shape[1], dtype=torch.bool, device=dev)
    for f in range(stack_n):
        apmask[f*frame + frame - 4 : f*frame + frame] = True
    xva0 = xva.clone(); xva0[:, apmask] = 0.0     # a_prev-blind val (reliance readout)

    best = float('inf'); t0 = time.time()
    for ep in range(1, a.epochs+1):
        net.train()
        tr_loss = n = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            if a.aprev_dropout > 0 or a.aprev_noise > 0:
                xb = xb.clone()
                if a.aprev_noise > 0:
                    xb[:, apmask] += a.aprev_noise * torch.randn(len(xb), int(apmask.sum()), device=dev)
                if a.aprev_dropout > 0:
                    drop = torch.rand(len(xb), device=dev) < a.aprev_dropout
                    xb[drop.unsqueeze(1) & apmask.unsqueeze(0)] = 0.0
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
            tr_loss += loss.item()*len(xb); n += len(xb)
        net.eval()
        with torch.no_grad():
            va_loss = lossf(net(xva), yva).item()
            va0_loss = lossf(net(xva0), yva).item()          # a_prev-blind
            reliance = (net(xva) - net(xva0)).abs().mean().item()
            # per-axis val RMSE in PHYSICAL units (denormalized by the caps)
            caps = torch.tensor(d['caps'], device=dev)
            rmse = (((net(xva)-yva)*caps)**2).mean(dim=0).sqrt().cpu().numpy()
        flag = ''
        if va_loss < best:
            best = va_loss; flag = '  *saved'
            torch.save({'state_dict': net.state_dict(),
                        'obs_dim': Xtr.shape[1], 'act_dim': 4,
                        'obs_names': list(d['obs_names']), 'caps': d['caps'].tolist(),
                        'stack_n': int(d['stack_n']), 'k_cal': float(d['k_cal']),
                        'val_mse': va_loss, 'aprev_dropout': a.aprev_dropout,
                        'aprev_noise': a.aprev_noise}, a.out)
        if ep % 2 == 0 or flag:
            print(f"ep {ep:3d}  train {tr_loss/n:.5f}  val {va_loss:.5f}  blind {va0_loss:.5f}  "
                  f"reliance {reliance:.4f}  "
                  f"RMSE[vx {rmse[0]:.3f} vy {rmse[1]:.3f} vz {rmse[2]:.3f} wz {rmse[3]:.3f}] m/s{flag}")
    print(f"\nbest val MSE {best:.5f}  ({time.time()-t0:.0f}s, {dev})  -> {a.out}")

if __name__ == '__main__':
    main()
