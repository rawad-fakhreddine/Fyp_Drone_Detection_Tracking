#!/usr/bin/env python3
"""build_deliverable_cal.py — assemble the calibrated-baseline deliverable from
the flat staged multiview clips (deliverable_s42/T{t}_C{c}.mp4) + csv_manifest.txt.
Per trajectory: a folder with C1+C2 mp4 + 3D-trajectory PNG + results .xlsx, plus
a master ALL_trajectories_summary.xlsx."""
import os,csv,shutil,subprocess,statistics as st
import pandas as pd
RES=os.path.expanduser("~/fyp/Results")
STAGE=f"{RES}/deliverable_s42"
HERE=os.path.dirname(os.path.abspath(__file__))
TRAJ={"1":"T1_static","2":"T2_slow","3":"T3_fast","4":"T4_orbit","5":"T5_lemniscate",
      "6":"T6_inclined_med","7":"T7_inclined_hard","8":"T8_helix"}
DISP={"1":"T1 static hover","2":"T2 slow straight","3":"T3 fast straight","4":"T4 circular orbit",
      "5":"T5 lemniscate","6":"T6 inclined medium","7":"T7 inclined hard","8":"T8 up-down helix"}

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
    first=next((i for i,r in enumerate(rows) if r['raw_det']=='REAL'),0)
    post=[r for r in rows[first:] if r.get('in_fov') in ('0','1')]
    cust=100*sum(1 for r in post if r.get('in_fov')=='1')/max(len(post),1)
    d=[fv(r,'true_dist_3d') for r in rows if fv(r,'true_dist_3d') is not None]
    hd=[fv(r,'true_dist_3d') for r in rows if r['phase']=='HOLD' and fv(r,'true_dist_3d') is not None]
    wz=[fv(r,'cmd_wz') for r in rows if r['phase']=='HOLD' and fv(r,'cmd_wz') is not None]
    yj=st.pstdev([abs(wz[i]-wz[i-1]) for i in range(1,len(wz))]) if len(wz)>2 else 0.0
    dur=fv(rows[-1],'sim_time')-fv(rows[0],'sim_time')
    return {"Detection %":round(det,1),"HOLD %":round(hold,1),"Custody %":round(cust,1),
            "In-FOV track %":round(trk,1),"Hold dist (m)":round(st.mean(hd),2) if hd else None,
            "Closest (m)":round(min(d),2) if d else None,
            "Yaw smoothness (jerk)":round(yj,4),"Duration (s)":round(dur)}

man={}
for line in open(f"{STAGE}/csv_manifest.txt"):
    p=line.split()
    if len(p)>=2: man[p[0]]=p[1]

master=[]
for t in "12345678":
    folder,disp=TRAJ[t],DISP[t]
    d=f"{STAGE}/{folder}"; os.makedirs(d,exist_ok=True)
    for c in ("1","2"):
        lbl=f"T{t}_C{c}"
        flat=f"{STAGE}/{lbl}.mp4"; csvp=man.get(lbl)
        if os.path.exists(flat): shutil.move(flat,f"{d}/{lbl}.mp4")
        if csvp and os.path.exists(csvp):
            png=f"{d}/{lbl}_3d.png"
            subprocess.run(["python3",f"{HERE}/make_3d_png.py",csvp,png,
                            f"{disp} — C{c} seed42 (calibrated k=0.077)"],check=True)
            master.append({"Trajectory":disp,"Config":f"C{c}",**metrics(csvp)})
            m=master[-1]
            print(f"  {disp:20s} C{c}  HOLD={m['HOLD %']}% cust={m['Custody %']}% det={m['Detection %']}% close={m['Closest (m)']}m hold={m['Hold dist (m)']}m")

dfm=pd.DataFrame(master)
for t in "12345678":
    folder,disp=TRAJ[t],DISP[t]
    sub=dfm[dfm['Trajectory']==disp]
    if len(sub):
        with pd.ExcelWriter(f"{STAGE}/{folder}/{folder}_results.xlsx",engine="xlsxwriter") as w:
            sub.to_excel(w,index=False,sheet_name=folder[:31])
with pd.ExcelWriter(f"{STAGE}/ALL_trajectories_summary.xlsx",engine="xlsxwriter") as w:
    dfm.to_excel(w,index=False,sheet_name="all_runs")
os.remove(f"{STAGE}/csv_manifest.txt") if os.path.exists(f"{STAGE}/csv_manifest.txt") else None
print("\nDeliverable ->",STAGE)
print(dfm.to_string(index=False))
