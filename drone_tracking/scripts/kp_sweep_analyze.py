#!/usr/bin/env python3
"""
kp_sweep_analyze.py — M10.3 Kp_wz x Kp_y sweep analyzer.

Reads each cell's per-seed flight CSV from ~/fyp/Results/diagnostics/kp_sweep/
and reports the per-cell table + verdict against the pre-stated accept gates:

  Cell vs base A, per seed {42,43,45}:
    * HOLD% lift on ALL three seeds                         (efficacy)
    * 3-seed-mean yaw zero-crossing rate < 0.21 Hz          (oscillation gate
        = baseline mean+2sd; >= 0.21 => OVER-GAIN, do NOT adopt, recommend
        Kp/Kd co-scaling follow-on)
    * no NEW saturation of cmd_wz (cap 0.50) / cmd_vy (cap 1.20)

  Outcomes:
    (a) lifts all 3 + xing OK + no new sat -> ADOPT (best cell), re-baseline
    (b) flat + not saturating             -> static tuning DONE (6 levers)
    (c) flat + now saturating             -> caps are the lever -> follow-on

Metrics per flight CSV (HOLD-phase, post-clamp/post-smooth published cmds):
    hold_pct, wz_sat%, vy_sat%, yaw_zerocross_hz, mean_recovery_s, aborted
All read-only. No adoption performed here — reports only.
"""
import os, csv, glob, argparse

ROOT = os.path.expanduser("~/fyp/Results/diagnostics/kp_sweep")
SEEDS = [42, 43, 45]
CELLS = ["A", "B", "C"]
GAINS = {"A": (0.9, 1.8), "B": (1.5, 1.8), "C": (1.5, 2.7)}
MAX_WZ, MAX_VY = 0.50, 1.20
SAT = 0.98
XING_GATE = 0.21          # baseline mean+2sd (pre-stated)
DUR = 200.0


def fnum(r, c):
    try:
        v = float(r.get(c, ""))
        return v
    except (ValueError, TypeError):
        return None


def analyze_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = len(rows)
    hold = [r for r in rows if r.get("phase", "") == "HOLD"]
    hold_pct = 100.0 * len(hold) / n
    # saturation over HOLD
    nw = nv = 0
    for r in hold:
        wz = fnum(r, "cmd_wz"); vy = fnum(r, "cmd_vy")
        if wz is not None and abs(wz) >= SAT * MAX_WZ: nw += 1
        if vy is not None and abs(vy) >= SAT * MAX_VY: nv += 1
    wz_sat = 100.0 * nw / len(hold) if hold else 0.0
    vy_sat = 100.0 * nv / len(hold) if hold else 0.0
    # yaw zero-crossing rate over HOLD
    def sgn(v): return 1 if v > 1e-4 else (-1 if v < -1e-4 else 0)
    x = 0; prev = None; t0 = t1 = None
    for r in hold:
        wz = fnum(r, "cmd_wz"); t = fnum(r, "sim_time")
        if wz is None or t is None: continue
        if t0 is None: t0 = t
        t1 = t
        s = sgn(wz)
        if prev is not None and s != 0 and prev != 0 and s != prev: x += 1
        if s != 0: prev = s
    span = (t1 - t0) if (t0 is not None and t1 is not None and t1 > t0) else 0.0
    xing = (x / span) if span > 5.0 else float("nan")
    # recovery: contiguous SEARCH episodes (closed only)
    eps = []; cs = cl = None
    for r in rows:
        if r.get("phase", "") == "SEARCH":
            t = fnum(r, "sim_time")
            if t is None: continue
            if cs is None: cs = t
            cl = t
        else:
            if cs is not None and cl is not None: eps.append(cl - cs)
            cs = cl = None
    rec = (sum(eps) / len(eps)) if eps else float("nan")
    # abort inference: logged sim span well short of requested duration
    times = [fnum(r, "sim_time") for r in rows if fnum(r, "sim_time") is not None]
    miss = (max(times) - min(times)) if times else 0.0
    aborted = 1 if miss < 0.9 * DUR else 0
    return dict(hold_pct=hold_pct, wz_sat=wz_sat, vy_sat=vy_sat,
                xing=xing, rec=rec, miss=miss, aborted=aborted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    data = {}
    for cell in CELLS:
        data[cell] = {}
        for s in SEEDS:
            p = os.path.join(a.root, cell, f"T7_s{s}_flight.csv")
            data[cell][s] = analyze_csv(p) if os.path.exists(p) else None

    print(f"\n=== M10.3 Kp_wz x Kp_y SWEEP — T7 Config2 Zone5, seeds {SEEDS} ===")
    print("(caps: max_wz=0.50 max_vy=1.20 | osc gate: 3-seed-mean xing >= 0.21 Hz = OVER-GAIN)\n")
    hdr = f"{'cell':4} {'Kp_wz':>5} {'Kp_y':>4} {'seed':>4} {'HOLD%':>6} {'wzSat%':>6} {'vySat%':>6} {'xingHz':>6} {'recS':>5} {'abort':>5}"
    print(hdr); print("-" * len(hdr))
    cell_mean = {}
    for cell in CELLS:
        kpwz, kpy = GAINS[cell]
        holds = []; xings = []
        for s in SEEDS:
            d = data[cell][s]
            if d is None:
                print(f"{cell:4} {kpwz:5.2f} {kpy:4.2f} {s:>4}    --- (no CSV)")
                continue
            holds.append(d["hold_pct"]);
            if d["xing"] == d["xing"]: xings.append(d["xing"])
            print(f"{cell:4} {kpwz:5.2f} {kpy:4.2f} {s:>4} {d['hold_pct']:6.1f} "
                  f"{d['wz_sat']:6.1f} {d['vy_sat']:6.1f} "
                  f"{(d['xing'] if d['xing']==d['xing'] else float('nan')):6.3f} "
                  f"{(d['rec'] if d['rec']==d['rec'] else 0):5.1f} {d['aborted']:>5}")
        mh = sum(holds)/len(holds) if holds else float('nan')
        mx = sum(xings)/len(xings) if xings else float('nan')
        cell_mean[cell] = dict(hold=mh, xing=mx, holds={s: (data[cell][s]['hold_pct'] if data[cell][s] else None) for s in SEEDS})
        print(f"{cell:4} {'':5} {'':4} {'MEAN':>4} {mh:6.1f} {'':6} {'':6} {mx:6.3f}")
        print("-" * len(hdr))

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n=== VERDICT (pre-stated gates) ===")
    base = cell_mean["A"]
    print(f"base A: mean HOLD%={base['hold']:.1f}, per-seed={ {s: round(v,1) if v is not None else None for s,v in base['holds'].items()} }, mean xing={base['xing']:.3f} Hz")
    for cell in ["B", "C"]:
        c = cell_mean[cell]
        if any(c['holds'][s] is None or base['holds'][s] is None for s in SEEDS):
            print(f"  {cell}: INCOMPLETE (missing seed CSVs) — cannot rule.")
            continue
        lifts_all = all(c['holds'][s] > base['holds'][s] for s in SEEDS)
        osc_ok = (c['xing'] == c['xing']) and c['xing'] < XING_GATE
        deltas = {s: round(c['holds'][s] - base['holds'][s], 1) for s in SEEDS}
        # new saturation: any seed cell pinning > ~1% where base ~0
        sat_new = False
        for s in SEEDS:
            d = data[cell][s]
            if d and (d['wz_sat'] > 1.0 or d['vy_sat'] > 2.0): sat_new = True
        tag = ("OVER-GAIN" if c['xing'] == c['xing'] and c['xing'] >= XING_GATE else "")
        print(f"  {cell}: ΔHOLD per-seed={deltas}  lifts_all={lifts_all}  "
              f"mean_xing={c['xing']:.3f}Hz(gate {XING_GATE}) osc_ok={osc_ok}  new_sat={sat_new} {tag}")
        if lifts_all and osc_ok and not sat_new:
            print(f"     -> WIN candidate (outcome a): adopt {cell} pending sign-off + re-baseline.")
        elif lifts_all and not osc_ok:
            print(f"     -> lifts HOLD% but OVER-GAIN (xing>=gate): do NOT adopt; "
                  f"follow-on = Kp/Kd co-scaling test.")
        elif not lifts_all and sat_new:
            print(f"     -> flat + saturating (outcome c): caps are the lever -> max_wz/max_vy follow-on.")
        else:
            print(f"     -> flat, not saturating (outcome b): no lift from this cell.")
    print("\n(If both B and C are flat & not saturating -> static tuning DONE, structural-FOV finding sealed across 6 levers.)")


if __name__ == "__main__":
    main()
