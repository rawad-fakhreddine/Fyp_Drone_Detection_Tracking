#!/usr/bin/env python3
"""make_3d_png.py CSV OUT.png [TITLE]
Clean 3D trajectory plot (target=blue, chaser=red) from a flight CSV.
Uses world-frame columns chaser_wx/wy/wz + target_wx/wy/wz."""
import sys,csv
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
def col(rows,k):
    out=[]
    for r in rows:
        try:out.append(float(r[k]))
        except:out.append(None)
    return out
def main():
    csvp,outp=sys.argv[1],sys.argv[2]
    title=sys.argv[3] if len(sys.argv)>3 else ""
    rows=[r for r in csv.DictReader(open(csvp)) if r.get('phase') in ('SEARCH','APPROACH','HOLD')]
    cx,cy,cz=col(rows,'chaser_wx'),col(rows,'chaser_wy'),col(rows,'chaser_wz')
    tx,ty,tz=col(rows,'target_wx'),col(rows,'target_wy'),col(rows,'target_wz')
    def clean(a,b,c):
        return zip(*[(x,y,z) for x,y,z in zip(a,b,c) if None not in (x,y,z)])
    cX,cY,cZ=clean(cx,cy,cz); tX,tY,tZ=clean(tx,ty,tz)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    ax.plot(tX,tY,tZ,color='#1f6fe0',lw=2.0,label='target')
    ax.plot(cX,cY,cZ,color='#d62728',lw=2.0,label='chaser')
    ax.scatter([tX[0]],[tY[0]],[tZ[0]],c='#1f6fe0',s=40,marker='o')
    ax.scatter([cX[0]],[cY[0]],[cZ[0]],c='#d62728',s=40,marker='o')
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left');ax.set_title(title)
    ax.view_init(elev=22,azim=-60)
    plt.tight_layout();plt.savefig(outp,dpi=130);print("wrote",outp)
main()
