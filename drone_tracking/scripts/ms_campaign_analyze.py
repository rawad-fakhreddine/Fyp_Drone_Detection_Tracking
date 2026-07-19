#!/usr/bin/env python3
"""ms_campaign_analyze.py — multi-seed C1-vs-C2 campaign analyzer (phase 2).

Reads the label->CSV mapping the batch tee'd into /tmp/ms_batch_results.txt
(lines: '=== ms seed=S cfg=C ===' followed by robust_run's final CSV path),
computes per-run metrics, prints a per-seed paired table + per-config
aggregates, and flags worst cases. Generalization-first: a parameter change is
only justified by a weakness that repeats ACROSS seeds, never by one flight.
"""
import csv, math, re, sys, statistics as st

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ms_batch_results.txt"

runs = []  # (seed, cfg, csv_path)
seed = cfg = None
for line in open(RESULTS):
    m = re.search(r"seed=(\d+)\s+cfg=(\d+)", line)
    if m:
        seed, cfg = int(m.group(1)), int(m.group(2))
    p = line.strip()
    if p.endswith(".csv") and seed is not None:
        runs.append((seed, cfg, p))
        seed = cfg = None

def metrics(path):
    try:
        rows = [r for r in csv.DictReader(open(path)) if r.get("true_dist_3d")]
    except FileNotFoundError:
        return None
    ap = [r for r in rows if r["phase"] in ("APPROACH", "HOLD")]
    if not ap:
        return None
    hold = [r for r in rows if r["phase"] == "HOLD"]
    d = [float(r["true_dist_3d"]) for r in ap]
    first = d[:200]
    out = dict(hold_pct=100.0 * len(hold) / len(ap),
               t_min=min(first), t_max=max(first),
               emerg=sum(1 for r in ap if r.get("emerg") == "1"),
               n=len(ap))
    det = [r for r in ap if r.get("raw_det")]
    out["det_pct"] = 100.0 * sum(1 for r in det if r["raw_det"] == "REAL") / max(len(det), 1)
    h = hold[len(hold) // 2:] if hold else []
    if h:
        hd = [float(r["true_dist_3d"]) for r in h]
        hv = [float(r["cmd_vx"]) for r in h]
        alt = [float(r["world_alt_err"]) for r in h if r.get("world_alt_err") not in (None, "", "nan")]
        out.update(dmean=st.mean(hd), dstd=st.pstdev(hd), vxstd=st.pstdev(hv),
                   alt=st.mean(alt) if alt else float("nan"))
    else:
        out.update(dmean=float("nan"), dstd=float("nan"), vxstd=float("nan"), alt=float("nan"))
    return out

print("%-5s %-4s | %6s %6s | %6s %6s %6s %6s | %5s %5s %5s" % (
    "seed", "cfg", "t_min", "t_max", "dmean", "dstd", "vxstd", "alt", "HOLD%", "det%", "emerg"))
agg = {1: [], 2: []}
for s, c, p in runs:
    m = metrics(p)
    if m is None:
        print("%-5d C%-3d | MISSING/EMPTY: %s" % (s, c, p)); continue
    print("%-5d C%-3d | %6.2f %6.2f | %6.2f %6.2f %6.3f %+6.2f | %5.0f %5.0f %5d" % (
        s, c, m["t_min"], m["t_max"], m["dmean"], m["dstd"], m["vxstd"], m["alt"],
        m["hold_pct"], m["det_pct"], m["emerg"]))
    agg[c].append(m)

for c in (1, 2):
    A = [m for m in agg[c] if not math.isnan(m["dstd"])]
    if not A:
        continue
    def ms(key):
        v = [m[key] for m in A]
        return "%.2f±%.2f" % (st.mean(v), st.pstdev(v))
    print("\nCONFIG %d (n=%d): t_min %s | t_max %s | dmean %s | dstd %s | vxstd %s | alt %s | HOLD%% %s | det%% %s | emerg_tot %d" % (
        c, len(A), ms("t_min"), ms("t_max"), ms("dmean"), ms("dstd"), ms("vxstd"),
        ms("alt"), ms("hold_pct"), ms("det_pct"), sum(m["emerg"] for m in A)))

# worst cases (deepest dip + biggest bounce) — the frames to READ
allr = [(s, c, p, metrics(p)) for s, c, p in runs]
allr = [x for x in allr if x[3]]
if allr:
    wd = min(allr, key=lambda x: x[3]["t_min"])
    wb = max(allr, key=lambda x: x[3]["t_max"])
    print("\nWORST dip:    seed %d C%d  t_min %.2f  (%s)" % (wd[0], wd[1], wd[3]["t_min"], wd[2]))
    print("WORST bounce: seed %d C%d  t_max %.2f  (%s)" % (wb[0], wb[1], wb[3]["t_max"], wb[2]))
