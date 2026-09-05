#!/usr/bin/env python3
# vz_sweep_analyze.py — M10.3 vertical Kp_z sweep analysis.
# Reads the cell->CSV mapping from vz_sweep's summary, then per (cell,traj):
#   HOLD% (3 seeds mean+-SD), vz saturation frac (HOLD), ey zero-crossing rate
#   (oscillation guard), abort count, mission duration. Applies the pre-stated
#   accept criterion (a/b/c) incl. the ey-oscillation hard gate vs baseline A.
import csv, glob, os, numpy as np
SUM=os.path.expanduser("~/fyp/Results/diagnostics/vz_sweep/sweep.log.summary")
GSUM=os.path.expanduser("~/fyp/Results/summary.csv")
CAP=1.5
def fnum(d,k):
    try: return float(d[k])
    except: return np.nan
# global summary.csv -> by csv basename (aborted, recovery, duration)
gmeta={}
if os.path.exists(GSUM):
    for r in csv.DictReader(open(GSUM)):
        gmeta[os.path.basename(r.get('csv_file',''))]=r
def metrics(csvpath):
    rows=list(csv.DictReader(open(csvpath)))
    ph=[x['phase'] for x in rows]
    t =np.array([fnum(x,'sim_time') for x in rows])
    vz=np.array([fnum(x,'cmd_vz') for x in rows])
    ey=np.array([fnum(x,'ey') for x in rows])
    hold=np.array([p=='HOLD' for p in ph])
    holdpct=100.0*np.mean(hold) if len(hold) else np.nan
    avz=np.abs(vz[hold]); avz=avz[~np.isnan(avz)]
    satf=100.0*np.mean(avz>=CAP-1e-3) if len(avz) else np.nan
    # ey zero-crossings within HOLD, per second of HOLD time
    th=t[hold]; eh=ey[hold]; m=~np.isnan(eh)
    th=th[m]; eh=eh[m]
    zc=0
    for i in range(1,len(eh)):
        if eh[i-1]!=0 and eh[i]!=0 and np.sign(eh[i])!=np.sign(eh[i-1]): zc+=1
    dur=(th[-1]-th[0]) if len(th)>1 else np.nan
    zchz=zc/dur if dur and dur>0 else np.nan
    return holdpct,satf,zchz
# parse cell mapping
recs=[]
for ln in open(SUM):
    p=ln.split()
    if not p or p[0]!='OK': continue
    cell=p[1]; traj=int(p[2][1:]); seed=int(p[3][1:])
    csvb=[x for x in p if x.startswith('csv=')][0][4:]
    cp=os.path.expanduser(f"~/fyp/Results/Config2/{csvb}")
    if not os.path.exists(cp): continue
    h,s,z=metrics(cp)
    g=gmeta.get(csvb,{})
    ab=int(float(g.get('aborted','0') or 0)) if g else 0
    rec=fnum(g,'mean_recovery_time_s') if g else np.nan
    mdur=fnum(g,'mission_duration_s') if g else np.nan
    recs.append(dict(cell=cell,traj=traj,seed=seed,hold=h,sat=s,zc=z,ab=ab,rec=rec,mdur=mdur))
if not recs:
    print("NO SWEEP RECORDS FOUND — has vz_sweep.sh completed?"); raise SystemExit
def agg(rs,k):
    v=np.array([r[k] for r in rs],float); v=v[~np.isnan(v)]
    return (np.mean(v),np.std(v)) if len(v) else (np.nan,np.nan)
print("="*72)
print(" M10.3 VERTICAL Kp_z SWEEP — PER-CELL TABLE  (Config 2, Zone 1)")
print("="*72)
KPZ={'A':3.0,'B':4.5,'C':6.0}
base={}  # (traj)-> baseline A hold list + zc mean for the guard
for traj in (6,7):
    print(f"\n--- T{traj} ---")
    print(f"  {'cell':5}{'Kp_z':6}{'HOLD% (mean±SD)':20}{'vzSat%':9}{'eyZC/s':9}{'abort':7}{'recov_s':9}")
    for cell in ('A','B','C'):
        rs=[r for r in recs if r['cell']==cell and r['traj']==traj]
        if not rs:
            print(f"  {cell:5}{KPZ[cell]:<6}  (no runs)"); continue
        hm,hs=agg(rs,'hold'); sm,_=agg(rs,'sat'); zm,_=agg(rs,'zc'); rm,_=agg(rs,'rec')
        nab=sum(r['ab'] for r in rs); holds=sorted(f"{r['hold']:.1f}" for r in rs)
        print(f"  {cell:5}{KPZ[cell]:<6}{hm:6.1f} ± {hs:4.1f} [{','.join(holds)}]  {sm:6.1f}  {zm:6.3f}  {nab}/{len(rs)}   {rm:6.1f}")
        if cell=='A': base[traj]=dict(holds=[r['hold'] for r in rs],zc=zm,seeds=[r['seed'] for r in rs])
# accept criterion
print("\n"+"="*72)
print(" PRE-STATED ACCEPT CRITERION")
print("="*72)
for traj in (6,7):
    if traj not in base: continue
    bA=base[traj]; aZ=bA['zc']
    bymap={r['seed']:r['hold'] for r in recs if r['cell']=='A' and r['traj']==traj}
    for cell in ('B','C'):
        rs=[r for r in recs if r['cell']==cell and r['traj']==traj]
        if len(rs)<3: continue
        allup=all(r['hold']>bymap.get(r['seed'],1e9) for r in rs)
        zc_cell,_=agg(rs,'zc'); osc = zc_cell > aZ*1.25 + 1e-9   # >25% over baseline = over-gain
        sat_cell,_=agg(rs,'sat'); newsat = sat_cell > 5.0
        tag = "WIN" if (allup and not osc and not newsat) else ("OSC-FAIL" if osc else ("SAT" if newsat else "flat/mixed"))
        print(f"  T{traj} {cell}: HOLD↑all3={allup}  eyZC {zc_cell:.3f} vs A {aZ:.3f} (osc={osc})  vzSat={sat_cell:.1f}% -> {tag}")
print("\n  Outcome key: (a) WIN on all-3 + no osc + no new sat -> adopt+re-baseline;")
print("               (b) flat & not saturating -> vertical channel CLOSED (detection-limited);")
print("               (c) flat but saturating -> lever is max_vz, run cell D.")
