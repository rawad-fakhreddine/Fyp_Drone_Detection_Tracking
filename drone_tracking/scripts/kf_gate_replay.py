#!/usr/bin/env python3
"""kf_gate_replay.py — OFFLINE confound-free screen of the Kalman measurement
gates PIXEL_JUMP_OUTLIER and MAX_CONSECUTIVE_REJECTIONS on the v4.1 detector.

Replays the recorded raw-YOLO stream (yolo_cx/cy/alpha_pix from a gt_projection
CSV == exactly what /drone_tracking/target_center published) through a byte-
faithful copy of kalman_filter_node's gate + KF, varying ONE parameter at a
time. Each REAL detection is labeled against GROUND TRUTH (proj_u/proj_v +
in_fov) as VALID (true target detection) or OUTLIER (false positive), so we can
score the gate's decisions directly:

  PIXEL_JUMP sweep:
    false_reject_rate = VALID detections the jump-gate rejected   (too low = bad)
    false_accept_rate = OUTLIER detections the gate accepted      (too high = bad)
  MAX_REJ sweep:
    post-escape recovery — after a >=MAX_REJ rejection streak forces acceptance,
    KF position error vs GT over the next K frames (re-lock vs corruption),
    plus the prediction-only frame burden (cost of waiting longer).

R/Q/damping held at node baseline. Nothing is flown or committed.
"""
import sys, csv, glob, math, os
import numpy as np

R_KEPT  = [6.0, 6.0, 5.0]
Q_BASE  = [0.5, 0.5, 3.0, 6.0, 6.0, 3.0]
MAXDROP = 30
# GT labeling cutoffs (px from the true projected center). Gap [VALID,OUTLIER]
# is ambiguous and excluded from rates. in_fov==0 + REAL => always OUTLIER.
VALID_PX   = 45.0
OUTLIER_PX = 100.0
K_POST     = 10          # frames scored after a forced acceptance


class KF:
    OPREV = 3000.0; ONEW = 900.0

    def __init__(self, pixel_jump, max_rej, damp=0.88):
        self.PJ = pixel_jump; self.MAXREJ = max_rej; self.damp = damp
        self.x = np.zeros(6); self.P = np.eye(6) * 500.0
        dt = 1.0/20.0
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.R = np.diag(R_KEPT); self.Q = np.diag(Q_BASE)
        self.init=False; self.drop=0; self.rej=0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        Kk = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kk @ y
        self.P = (np.eye(6) - Kk @ self.H) @ self.P

    def _outlier(self, mx, my, mz):
        prev = self.x[2]; jump = math.hypot(mx - self.x[0], my - self.x[1])
        if prev > self.OPREV and mz < self.ONEW:
            return True, jump
        if jump > self.PJ and self.rej < self.MAXREJ:
            return True, jump
        return False, jump

    def step(self, mx, my, mz):
        """Mirror node callback. Returns ('INIT'|'ACCEPT'|'REJECT'|'ESCAPE', jump)."""
        if mx is None:                       # dropout (NaN published)
            self.drop += 1
            if self.init:
                self.x[3]*=self.damp; self.x[4]*=self.damp; self.x[5]*=self.damp
                if self.drop >= MAXDROP: self.x[3]=self.x[4]=self.x[5]=0.0
                self.predict()
            self._clip(); return 'DROP', 0.0
        if not self.init:
            self.x[0:3]=[mx,my,mz]; self.x[3:6]=0.0
            self.init=True; self.drop=0; self.rej=0
            self.predict(); self.update(np.array([mx,my,mz])); self._clip()
            return 'INIT', 0.0
        outlier, jump = self._outlier(mx, my, mz)
        if outlier and self.rej >= self.MAXREJ:          # alpha escape hatch
            self.x[2]=mz; self.x[5]=0.0; self.P[2,2]=500.0; self.P[5,5]=500.0
            self.rej=0; self.drop=0
            self.predict(); self.update(np.array([mx,my,mz])); self._clip()
            return 'ESCAPE', jump
        if outlier:
            self.rej+=1; self.drop+=1
            self.x[3]*=self.damp; self.x[4]*=self.damp; self.x[5]*=self.damp
            if self.drop>=MAXDROP: self.x[3]=self.x[4]=self.x[5]=0.0
            self.predict(); self._clip()
            return 'REJECT', jump
        # accept (normal OR pos-guard re-admit after >=MAXREJ)
        guard = self.rej >= self.MAXREJ
        self.drop=0; self.rej=0
        self.predict(); self.update(np.array([mx,my,mz])); self._clip()
        return ('GUARD' if guard else 'ACCEPT'), jump

    def _clip(self):
        self.x[0]=np.clip(self.x[0],0.,640.); self.x[1]=np.clip(self.x[1],0.,480.)
        self.x[2]=np.clip(self.x[2],0.,307200.)


def fnum(s):
    try:
        v=float(s); return v if v==v else None
    except (ValueError, TypeError):
        return None


def load(path):
    rows=[]
    for r in csv.DictReader(open(path)):
        rows.append(dict(
            in_fov = r.get('in_fov')=='1',
            pu = fnum(r.get('proj_u')), pv = fnum(r.get('proj_v')),
            ycx = fnum(r.get('yolo_cx')), ycy = fnum(r.get('yolo_cy')),
            yal = fnum(r.get('yolo_alpha_pix')),
            real = r.get('yolo_det')=='REAL'))
    return rows


def label(row):
    """VALID / OUTLIER / AMBIG for a REAL detection, vs ground truth."""
    if not row['in_fov']:
        return 'OUTLIER'                       # target not in frame => false pos
    if row['pu'] is None or row['ycx'] is None:
        return 'AMBIG'
    d = math.hypot(row['ycx']-row['pu'], row['ycy']-row['pv'])
    if d <= VALID_PX:   return 'VALID'
    if d >  OUTLIER_PX: return 'OUTLIER'
    return 'AMBIG'


def run_stream(rows, pixel_jump, max_rej):
    kf = KF(pixel_jump, max_rej)
    fr=fa=nv=no=0                              # false-reject/accept, n valid/outlier
    events=[]                                  # forced-accept (post >=MAXREJ) events
    streak=0
    perr=[]                                    # per-frame |kf-gt| on in_fov frames
    hist=[]                                    # (kf_xy, gt_xy or None) timeline
    for row in rows:
        if row['real'] and row['ycx'] is not None:
            mx,my,mz = row['ycx'], row['ycy'], (row['yal'] or 0.0)
            lab = label(row)
        else:
            mx=my=mz=None; lab=None
        dec, jump = kf.step(mx,my,mz)
        # score gate decision against GT label
        if lab in ('VALID','OUTLIER'):
            accepted = dec in ('ACCEPT','GUARD','ESCAPE','INIT')
            if lab=='VALID':
                nv+=1
                if not accepted: fr+=1
            else:
                no+=1
                if accepted: fa+=1
        # rejection streak bookkeeping for MAX_REJ analysis
        if dec=='REJECT':
            streak+=1
        elif dec in ('ACCEPT','GUARD','ESCAPE'):
            if streak>=max_rej:
                events.append(len(hist))       # index of this accepted frame
            streak=0
        # GT error timeline (in_fov only, where truth pixel is meaningful)
        gt = (row['pu'],row['pv']) if (row['in_fov'] and row['pu'] is not None) else None
        hist.append(((kf.x[0],kf.x[1]), gt))
        if gt is not None:
            perr.append(math.hypot(kf.x[0]-gt[0], kf.x[1]-gt[1]))
    # post-escape recovery: KF error over K frames after each forced acceptance
    post=[]; corrupt=0
    for idx in events:
        errs=[]
        for j in range(idx, min(idx+K_POST, len(hist))):
            kfxy,gt = hist[j]
            if gt is not None:
                errs.append(math.hypot(kfxy[0]-gt[0], kfxy[1]-gt[1]))
        if errs:
            post.append(sum(errs)/len(errs))
            # corruption = error grows over the window (end worse than start by >20px)
            if len(errs)>=4 and (errs[-1]-errs[0])>20.0: corrupt+=1
    return dict(fr=fr,fa=fa,nv=nv,no=no, n_events=len(events),
                post=post, corrupt=corrupt,
                rej_frames=sum(1 for h in [None] for _ in []),  # placeholder
                mean_err=(sum(perr)/len(perr)) if perr else float('nan'),
                n_reject=streak)  # not used


def pool(files, pixel_jump, max_rej):
    agg=dict(fr=0,fa=0,nv=0,no=0,n_events=0,corrupt=0,post=[],errs=[])
    for f in files:
        rows=load(f)
        r=run_stream(rows, pixel_jump, max_rej)
        for k in ('fr','fa','nv','no','n_events','corrupt'): agg[k]+=r[k]
        agg['post']+=r['post']
        if r['mean_err']==r['mean_err']: agg['errs'].append(r['mean_err'])
    agg['frr']=100.0*agg['fr']/agg['nv'] if agg['nv'] else float('nan')
    agg['far']=100.0*agg['fa']/agg['no'] if agg['no'] else float('nan')
    agg['post_mean']=(sum(agg['post'])/len(agg['post'])) if agg['post'] else float('nan')
    agg['mean_err']=(sum(agg['errs'])/len(agg['errs'])) if agg['errs'] else float('nan')
    return agg


def main():
    base = os.path.expanduser('~/fyp/Results/diagnostics')
    t7 = sorted(glob.glob(base+'/search_latch_v630/T7_*_gt.csv') +
                glob.glob(base+'/search_tau/tau*/T7_*_gt.csv'))
    t9 = sorted(glob.glob(base+'/search_latch_v630/T9_*_gt.csv') +
                glob.glob(base+'/search_tau/tau*/T9_*_gt.csv'))
    allf = t7+t9
    print("streams: T7=%d  T9=%d  total=%d" % (len(t7),len(t9),len(allf)))
    print("labels: VALID<=%.0fpx, OUTLIER>%.0fpx from GT proj (in_fov=0 REAL=>OUTLIER); K_post=%d\n"
          % (VALID_PX, OUTLIER_PX, K_POST))

    print("="*74)
    print("TABLE 1 — PIXEL_JUMP sweep   (MAX_REJ=4 fixed)   pooled ALL T7+T9")
    print("="*74)
    print("%8s %12s %12s %10s %12s" %
          ("PIX_JUMP","false_rej%","false_acc%","n_valid","n_outlier"))
    print("-"*74)
    for pj in [120,150,180,220,260]:
        a=pool(allf, pj, 4)
        mark = "  <-- current" if pj==180 else ""
        print("%8d %11.2f%% %11.2f%% %10d %12d%s" %
              (pj, a['frr'], a['far'], a['nv'], a['no'], mark))
    print("\n  (per-traj split at the recommended/current threshold)")
    for pj in [150,180]:
        a7=pool(t7,pj,4); a9=pool(t9,pj,4)
        print("   PJ=%d  T7 fr%%=%.2f fa%%=%.2f | T9 fr%%=%.2f fa%%=%.2f"
              % (pj, a7['frr'],a7['far'], a9['frr'],a9['far']))

    print("\n"+"="*74)
    print("TABLE 2 — MAX_REJ sweep   (PIXEL_JUMP=180 fixed)   pooled ALL T7+T9")
    print("="*74)
    print("%8s %10s %16s %12s %12s" %
          ("MAX_REJ","n_escape","post_err_px(K=10)","corrupt","track_err_px"))
    print("-"*74)
    for mr in [2,3,4,6]:
        a=pool(allf, 180, mr)
        mark="  <-- current" if mr==4 else ""
        cf = (100.0*a['corrupt']/a['n_events']) if a['n_events'] else float('nan')
        print("%8d %10d %16.1f %9.0f%%(%d) %12.1f%s" %
              (mr, a['n_events'], a['post_mean'], cf, a['corrupt'], a['mean_err'], mark))
    print("\n  post_err = mean |KF-GT| over 10 frames after each forced acceptance")
    print("  corrupt  = forced-accepts whose error GREW >20px across the window")
    print("  track_err= mean |KF-GT| over all in_fov frames (net tracking quality)")


if __name__ == '__main__':
    main()
