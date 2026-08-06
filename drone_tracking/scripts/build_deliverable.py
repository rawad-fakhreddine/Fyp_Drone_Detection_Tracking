#!/usr/bin/env python3
"""build_deliverable.py — assemble the supervisor-facing multiview deliverable.
Per trajectory: a folder with C1+C2 mp4 + 3D-trajectory PNG + results .xlsx.
Plus a master ALL_trajectories_summary.xlsx. Reads world-frame flight CSVs."""
import os,csv,glob,shutil,subprocess,sys
import pandas as pd
RES=os.path.expanduser("~/fyp/Results")
KEEP=f"{RES}/multiview/all_C1C2_s42"
OUT=f"{RES}/deliverable_s42"
HERE=os.path.dirname(os.path.abspath(__file__))

# trajectory display names + the source flight CSV per (clip)
TRAJ={
 "1":("T1_static", "T1 static hover"),
 "2":("T2_slow",   "T2 slow straight"),
 "3":("T3_fast",   "T3 fast straight"),
 "4":("T4_orbit",  "T4 circular orbit"),
 "5":("T5_lemniscate","T5 lemniscate"),
 "6":("T6_inclined_med","T6 inclined medium"),
 "7":("T7_inclined_hard","T7 inclined hard"),
 "8":("T8_helix",  "T8 up-down helix"),
}
# clip label -> (mp4 in KEEP, source CSV glob under RES).  NEWEST => pick latest.
CLIPS=[
 # traj, label,       mp4 name,                 csv glob
 ("1","C1", "T1_C1_s42.mp4", "Config1/traj1_zone1_2026-08-03_18-03-23.csv"),
 ("1","C2", "T1_C2_s42.mp4", "Config2/traj1_zone1_2026-08-03_18-09-14.csv"),
 ("2","C1", "T2_C1_s42.mp4", "Config1/traj2_zone1_2026-08-03_18-44-32.csv"),
 ("2","C2", "T2_C2_s42.mp4", "Config2/traj2_zone1_2026-08-03_18-50-25.csv"),
 ("3","C1", "T3_C1_s42.mp4", "NEWEST:Config1/traj3_zone1_2026-08-03_*.csv"),
 ("3","C2", "T3_C2_s42.mp4", "NEWEST:Config2/traj3_zone1_2026-08-03_*.csv"),
 ("4","C1", "T4_C1_s42.mp4", "Config1/traj4_zone1_2026-08-03_17-51-31.csv"),
 ("4","C2", "T4_C2_s42.mp4", "Config2/traj4_zone1_2026-08-03_17-57-28.csv"),
 ("5","C1", "T5_C1_s42.mp4", "Config1/traj5_zone1_2026-08-03_16-42-29.csv"),
 ("5","C2", "T5_C2_s42.mp4", "Config2/traj5_zone1_2026-08-03_16-48-23.csv"),
 ("6","C1", "T6_C1_s42.mp4", "Config1/traj6_zone1_2026-08-03_16-08-23.csv"),
 ("6","C2", "T6_C2_s42.mp4", "Config2/traj6_zone1_2026-08-03_16-14-19.csv"),
 ("7","C1", "T7_C1_s42.mp4", "Config1/traj7_zone1_2026-08-02_22-40-12.csv"),
 ("7","C2", "T7_C2_s42.mp4", "Config2/traj7_zone1_2026-08-02_22-45-31.csv"),
 ("7","C1_fwdzigzag","T7_fwdzigzag_C1_s42.mp4","Config1/traj7_zone1_2026-08-02_23-08-51.csv"),
 ("8","C1", "T8_C1_s42.mp4", "Config1/traj8_zone1_2026-08-03_16-28-20.csv"),
 ("8","C2", "T8_C2_s42.mp4", "Config2/traj8_zone1_2026-08-03_16-34-21.csv"),
]

def resolve(g):
    if g.startswith("NEWEST:"):
        cands=sorted(glob.glob(f"{RES}/{g[7:]}"),key=os.path.getmtime)
        return cands[-1] if cands else None
    p=f"{RES}/{g}"; return p if os.path.exists(p) else None

def metrics(csvp):
    rows=[r for r in csv.DictReader(open(csvp)) if r.get('phase') in ('SEARCH','APPROACH','HOLD')]
    def fv(r,k):
        try:return float(r[k])
        except:return None
    fov=[r for r in rows if r.get('in_fov')=='1']
    flg=[r for r in rows if r.get('in_fov') in ('0','1')]
    det=100*sum(1 for r in fov if r['raw_det']=='REAL')/max(len(fov),1)
    hold=100*sum(1 for r in rows if r['phase']=='HOLD')/max(len(rows),1)
    trk=100*len(fov)/max(len(flg),1)
    # custody = target-in-FOV fraction AFTER first real detection
    first=next((i for i,r in enumerate(rows) if r['raw_det']=='REAL'),0)
    post=[r for r in rows[first:] if r.get('in_fov') in ('0','1')]
    cust=100*sum(1 for r in post if r.get('in_fov')=='1')/max(len(post),1)
    d=[fv(r,'true_dist_3d') for r in rows if fv(r,'true_dist_3d') is not None]
    hd=[fv(r,'true_dist_3d') for r in rows if r['phase']=='HOLD' and fv(r,'true_dist_3d') is not None]
    # yaw jerk = std of consecutive cmd_wz diffs during HOLD (smoothness proxy)
    wz=[fv(r,'cmd_wz') for r in rows if r['phase']=='HOLD' and fv(r,'cmd_wz') is not None]
    import statistics as st
    yj=st.pstdev([abs(wz[i]-wz[i-1]) for i in range(1,len(wz))]) if len(wz)>2 else 0.0
    dur=fv(rows[-1],'sim_time')-fv(rows[0],'sim_time')
    return {
      "Detection %":round(det,1),"HOLD %":round(hold,1),"Custody %":round(cust,1),
      "In-FOV track %":round(trk,1),
      "Hold dist (m)":round(st.mean(hd),2) if hd else None,
      "Closest (m)":round(min(d),2) if d else None,
      "Yaw smoothness (jerk)":round(yj,4),
      "Duration (s)":round(dur),
    }

def main():
    os.makedirs(OUT,exist_ok=True)
    master=[]
    for traj,lbl,mp4,cg in CLIPS:
        folder,disp=TRAJ[traj]
        d=f"{OUT}/{folder}";os.makedirs(d,exist_ok=True)
        src_mp4=f"{KEEP}/{mp4}"; csvp=resolve(cg)
        if not os.path.exists(src_mp4): print("  MISSING mp4:",mp4);continue
        if not csvp: print("  MISSING csv:",cg);continue
        shutil.copy(src_mp4,f"{d}/{mp4}")
        png=f"{d}/{mp4.replace('.mp4','_3d.png')}"
        subprocess.run(["python3",f"{HERE}/make_3d_png.py",csvp,png,f"{disp} — {lbl} seed42"],check=True)
        m=metrics(csvp); m={"Trajectory":disp,"Run":lbl,**m}
        master.append(m)
        print(f"  {disp:22s} {lbl:12s} HOLD={m['HOLD %']}% cust={m['Custody %']}% det={m['Detection %']}% close={m['Closest (m)']}m")
    # per-trajectory xlsx + master
    dfm=pd.DataFrame(master)
    for traj,(folder,disp) in TRAJ.items():
        sub=dfm[dfm['Trajectory']==disp]
        if len(sub):
            with pd.ExcelWriter(f"{OUT}/{folder}/{folder}_results.xlsx",engine="xlsxwriter") as w:
                sub.to_excel(w,index=False,sheet_name=folder[:31])
    with pd.ExcelWriter(f"{OUT}/ALL_trajectories_summary.xlsx",engine="xlsxwriter") as w:
        dfm.to_excel(w,index=False,sheet_name="all_runs")
    print("\nDeliverable ->",OUT)
    print(dfm.to_string(index=False))

main()
