#!/usr/bin/env python3
"""plot_both_paths.py CSV [OUT_PREFIX] — plot the TARGET and CHASER world paths
together (3-D), export a clean positions CSV, and print separation stats. Uses
flight_logger's target_wx/wy/wz + chaser_wx/wy/wz. Handy for the random path
(T10) to see how the chaser follows the target."""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/both_paths'

rows = [r for r in csv.DictReader(open(CSV))
        if r.get('target_wx') not in (None, '', 'nan')
        and r.get('chaser_wx') not in (None, '', 'nan')]
tx = np.array([float(r['target_wx']) for r in rows])
ty = np.array([float(r['target_wy']) for r in rows])
tz = np.array([float(r['target_wz']) for r in rows])
cx = np.array([float(r['chaser_wx']) for r in rows])
cy = np.array([float(r['chaser_wy']) for r in rows])
cz = np.array([float(r['chaser_wz']) for r in rows])
ts = np.array([float(r['sim_time']) for r in rows])
i0 = next((i for i, z in enumerate(tz) if z > 12.5), 0) + 50   # MOVING segment
sl = slice(i0, None)
tx, ty, tz, cx, cy, cz, ts = (a[sl] for a in (tx, ty, tz, cx, cy, cz, ts))
sep = np.sqrt((tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2)

# --- positions CSV ---
with open(OUT + '_positions.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sim_time', 'target_x', 'target_y', 'target_z',
                'chaser_x', 'chaser_y', 'chaser_z', 'separation_m'])
    for i in range(len(tx)):
        w.writerow(['%.2f' % ts[i], '%.3f' % tx[i], '%.3f' % ty[i], '%.3f' % tz[i],
                    '%.3f' % cx[i], '%.3f' % cy[i], '%.3f' % cz[i], '%.3f' % sep[i]])

# --- 3-D plot ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(tx, ty, tz, color='#1f77b4', lw=1.3, label='TARGET')
ax.plot(cx, cy, cz, color='#d62728', lw=1.3, label='CHASER')
ax.scatter([tx[0]], [ty[0]], [tz[0]], color='#1f77b4', s=45, marker='o')
ax.scatter([cx[0]], [cy[0]], [cz[0]], color='#d62728', s=45, marker='o')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.view_init(elev=22, azim=-58)
rx, ry, rz = np.ptp(np.r_[tx, cx]), np.ptp(np.r_[ty, cy]), np.ptp(np.r_[tz, cz])
m = max(rx, ry, rz, 1e-3)
ax.set_box_aspect((rx / m, ry / m, max(rz / m, 0.15)))
ax.legend(loc='upper left', fontsize=10)
ax.set_title('Target vs Chaser world paths\nseparation: mean %.2f m  min %.2f m  max %.2f m'
             % (sep.mean(), sep.min(), sep.max()), fontsize=11)
fig.tight_layout()
fig.savefig(OUT + '_3d.png', dpi=130)
plt.close(fig)

print('samples %d (%.0f s) | separation mean %.2f m  min %.2f m  max %.2f m'
      % (len(tx), ts[-1] - ts[0], sep.mean(), sep.min(), sep.max()))
print('-> %s_3d.png' % OUT)
print('-> %s_positions.csv' % OUT)
