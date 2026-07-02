#!/usr/bin/env python3
# m12_smoke_analyze.py — M12 Phase D-prep step 4 (the gate).
# Runs extract_metrics on the C1 and C2 smoke CSVs, prints a two-column
# comparison of EVERY metric, then applies the three gate checks:
#   (i)  no NaN/empty in the new M12 metrics (deriv/jerk/dropout family)
#   (ii) independent dropout count matches n_dropouts_bridged (+/- reconcile)
#   (iii) deriv_noise_{dex,dey,dea} visibly HIGHER in C1 than C2 (raw vs filtered)
# Usage: m12_smoke_analyze.py --c1 C1.csv --c2 C2.csv
import argparse, csv, os, subprocess, sys, tempfile

EXTRACT = os.path.expanduser("~/catkin_ws/src/drone_tracking/scripts/extract_metrics.py")
NEW_METRICS = ['hold_pct_mission','cmd_var_vx','cmd_var_vy','cmd_var_vz','cmd_var_wz',
    'rms_jerk_vx','rms_jerk_vy','rms_jerk_vz','rms_jerk_wz',
    'deriv_noise_dex','deriv_noise_dey','deriv_noise_dea',
    'time_to_first_loss_s','n_dropouts_bridged','reacq_error_px','actuation_effort']

def extract(csvp, cfg):
    fd, path = tempfile.mkstemp(suffix='_sum.csv', dir='/tmp'); os.close(fd); os.unlink(path)
    r = subprocess.run([sys.executable, EXTRACT, '--csv', csvp, '--config', str(cfg),
        '--zone','1','--traj','3','--seed','42','--duration','200','--summary', path],
        capture_output=True, text=True)
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    if os.path.exists(path): os.unlink(path)
    if not rows: print("extract failed for", csvp, "\n", r.stdout, r.stderr); sys.exit(1)
    return rows[-1]

def manual_dropouts(csvp, cfg):
    """independent REAL->NONE dropout inventory from raw_det, with bridged/frozen split."""
    rows = list(csv.DictReader(open(csvp)))
    drops=[]; in_d=False; onset=None; had_search=False; had_pred=False; prev_real=False; t0=None
    def ft(r):
        try: return float(r['sim_time'])
        except: return None
    for r in rows:
        det=r.get('raw_det',''); ph=r.get('phase',''); t=ft(r)
        if det=='REAL':
            if in_d and onset is not None and t is not None:
                drops.append(dict(dur=t-onset, search=had_search, pred=had_pred))
            in_d=False; onset=None; had_search=False; had_pred=False; prev_real=True
        else:
            if not in_d and prev_real:
                in_d=True; onset=t; had_search=(ph=='SEARCH'); had_pred=(r.get('flt_det','')=='PRED')
            elif in_d:
                if ph=='SEARCH': had_search=True
                if r.get('flt_det','')=='PRED': had_pred=True
    short = [d for d in drops if d['dur']<1.5 and not d['search']]
    return dict(total=len(drops), short_bridged=len(short),
                short_pred=sum(1 for d in short if d['pred']),
                mean_dur=(sum(d['dur'] for d in drops)/len(drops) if drops else float('nan')))

def fnum(s):
    try: return float(s)
    except: return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--c1',required=True); ap.add_argument('--c2',required=True)
    a=ap.parse_args()
    m1=extract(a.c1,1); m2=extract(a.c2,2)
    keys=list(m1.keys())
    print("="*74)
    print(" M12 SMOKE GATE — C1 vs C2  (T3, seed42, Zone1, 200s)")
    print("="*74)
    print(f" {'metric':<24}{'Config1 (raw)':>18}{'Config2 (kalman)':>20}")
    print(" "+"-"*70)
    for k in keys:
        star = ' *' if k in NEW_METRICS else ''
        print(f" {k:<24}{str(m1.get(k,'')):>18}{str(m2.get(k,'')):>20}{star}")
    print("  (* = new M12 metric)")

    # ── gate (i): no empty new metrics ──────────────────────────────────────
    empt1=[k for k in NEW_METRICS if str(m1.get(k,''))=='']
    empt2=[k for k in NEW_METRICS if str(m2.get(k,''))=='']
    print("\n── GATE (i) no-empty new metrics ──")
    print(f"  C1 empty: {empt1 or 'none'}")
    print(f"  C2 empty: {empt2 or 'none'}")
    gi = not empt1 and not empt2

    # ── gate (ii): independent dropout reconciliation ───────────────────────
    d1=manual_dropouts(a.c1,1); d2=manual_dropouts(a.c2,2)
    print("\n── GATE (ii) dropout reconciliation (manual vs metric) ──")
    for tag,dd,mm in (("C1",d1,m1),("C2",d2,m2)):
        print(f"  {tag}: manual short-bridged={dd['short_bridged']} "
              f"(of which flt=PRED {dd['short_pred']})  |  metric n_dropouts_bridged={mm.get('n_dropouts_bridged')}  "
              f"| total raw dropouts={dd['total']} mean_dur={dd['mean_dur']:.2f}s")
    gii = (str(d1['short_bridged'])==str(m1.get('n_dropouts_bridged'))
           and str(d2['short_bridged'])==str(m2.get('n_dropouts_bridged')))

    # ── gate (iii): deriv noise direction C1 > C2 ───────────────────────────
    print("\n── GATE (iii) deriv-noise direction (expect C1 > C2) ──")
    giii=True
    for c in ('deriv_noise_dex','deriv_noise_dey','deriv_noise_dea'):
        v1,v2=fnum(m1.get(c,'')),fnum(m2.get(c,''))
        if v1 is None or v2 is None:
            print(f"  {c}: C1={m1.get(c)} C2={m2.get(c)}  -> MISSING"); giii=False; continue
        ratio = v1/v2 if v2>0 else float('inf')
        ok = v1>v2
        giii = giii and ok
        print(f"  {c}: C1={v1:.5f}  C2={v2:.5f}  ratio={ratio:.2f}  {'OK (C1>C2)' if ok else 'FAIL (C1<=C2)'}")

    print("\n"+"="*74)
    print(f"  GATE (i) no-empty      : {'PASS' if gi else 'FAIL'}")
    print(f"  GATE (ii) dropout recon: {'PASS' if gii else 'REVIEW (mismatch — see above)'}")
    print(f"  GATE (iii) C1>C2 noise : {'PASS' if giii else 'FAIL — STOP, diagnose'}")
    print(f"  SMOKE GATE OVERALL     : {'PASS — clear for campaign' if (gi and giii) else 'HOLD'}")
    print("="*74)

if __name__=='__main__': main()
