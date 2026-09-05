#!/usr/bin/env python3
# vz_precheck.py — M10.3 vertical-channel pre-check (read-only, no flying).
# From the M11 Config-2 Zone-1 baselines T6/T7 x seeds{42,43,45}:
#   (A) vz saturation: fraction of HOLD cycles where |cmd_vz| hits max_vz=1.5
#       (mean/p90/p99 of |cmd_vz|, frac at cap). Decides gain vs cap lever.
#   (B) ey-drift: in the 2s window before each loss onset (raw_det REAL->NONE),
#       is ey ramping toward the frame edge (lag, Kp_z can help) or flat near
#       centre (pure detection, gain can't help)?
import glob, numpy as np, csv
CAP=1.5; WIN=2.0; EDGE=0.8   # max_vz, pre-loss window s, |ey| clip = frame-edge proxy
def load(f):
    r=list(csv.DictReader(open(f)))
    return r
def fnum(r,k):
    try: return float(r[k])
    except: return np.nan
print(f"{'='*68}\n M10.3 VERTICAL-CHANNEL PRE-CHECK  (max_vz cap={CAP})\n{'='*68}")
sat_all=[]; drift_rows=[]
for traj in (6,7):
    # CSV names carry no seed -> take the 3 NEWEST distinct files = the M11 matrix
    # Config-2 cells (3 seeds). Label them r1/r2/r3 (newest first).
    fs=sorted(glob.glob(f"/home/rawad/fyp/Results/Config2/traj{traj}_zone1_*.csv"),reverse=True)[:3]
    for ridx,f in enumerate(fs,1):
        seed=f"r{ridx}"; rows=load(f)
        ph =[x['phase'] for x in rows]
        t  =np.array([fnum(x,'sim_time') for x in rows])
        vz =np.array([fnum(x,'cmd_vz') for x in rows])
        ey =np.array([fnum(x,'ey') for x in rows])
        det=[x.get('raw_det','') for x in rows]
        hold=np.array([p=='HOLD' for p in ph])
        avz=np.abs(vz[hold]); avz=avz[~np.isnan(avz)]
        if len(avz):
            frac=float(np.mean(avz>=CAP-1e-3))
            sat_all.append(avz)
            print(f"  T{traj} s{seed}: HOLD n={len(avz):4d} | "
                  f"mean|vz|={np.mean(avz):.3f} p90={np.percentile(avz,90):.3f} "
                  f"p99={np.percentile(avz,99):.3f} | frac@cap={frac*100:.1f}%")
        # loss onsets: raw_det REAL -> NONE
        for i in range(1,len(det)):
            if det[i-1]=='REAL' and det[i]=='NONE':
                t0=t[i]; m=(t>=t0-WIN)&(t<t0)
                eyw=ey[m]; tw=t[m]; eyw=eyw[~np.isnan(eyw)]
                if len(eyw)>=4:
                    sl=np.polyfit(tw[~np.isnan(ey[m])][:len(eyw)],eyw,1)[0] if len(tw)==len(eyw) else np.polyfit(np.arange(len(eyw)),eyw,1)[0]
                    drift_rows.append((traj,seed,abs(eyw[0]),abs(eyw[-1]),
                                       np.mean(np.abs(eyw)),sl))
print(f"\n--- (A) SATURATION VERDICT ---")
if sat_all:
    allv=np.concatenate(sat_all)
    fr=float(np.mean(allv>=CAP-1e-3))
    print(f"  pooled HOLD |cmd_vz|: mean={np.mean(allv):.3f}  p90={np.percentile(allv,90):.3f}  "
          f"p99={np.percentile(allv,99):.3f}  max={allv.max():.3f}")
    print(f"  pooled frac at cap (|vz|>={CAP}): {fr*100:.2f}%   cap={CAP}")
    print(f"  => {'ALREADY SATURATING (lever=max_vz, cell D)' if fr>0.05 else 'NOT saturating -> Kp_z has headroom, gain sweep VALID'}")
print(f"\n--- (B) ey-DRIFT BEFORE LOSS ONSET (raw REAL->NONE, {WIN}s window) ---")
if drift_rows:
    print(f"  {'traj':5}{'seed':5}{'|ey|start':10}{'|ey|onset':10}{'mean|ey|':10}{'slope/s':10}")
    rise=0
    for (tr,sd,e0,e1,em,sl) in drift_rows:
        print(f"  T{tr:<4}{sd:<5}{e0:<10.3f}{e1:<10.3f}{em:<10.3f}{sl:<+10.3f}")
        if e1>e0 and abs(e1)>0.30: rise+=1
    n=len(drift_rows)
    print(f"\n  onsets={n} | edge-ward ramp (|ey|onset>start & >0.30): {rise}/{n} = {rise/n*100:.0f}%")
    print(f"  mean |ey| at onset = {np.mean([r[3] for r in drift_rows]):.3f}  "
          f"(EDGE proxy={EDGE}); mean slope = {np.mean([r[5] for r in drift_rows]):+.3f}/s")
    print(f"  => {'LAG-dominated (ey ramps to edge) -> Kp_z plausibly helps' if rise>n/2 else 'NEAR-CENTRE at drop -> pure detection, gain unlikely to help'}")
else:
    print("  no REAL->NONE onsets found in HOLD-region windows")
