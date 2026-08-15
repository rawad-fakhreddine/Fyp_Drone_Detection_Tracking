#!/usr/bin/env python3
"""make_3d_png.py CSV OUT.png [TITLE]
4-view trajectory panel (target=blue, chaser=red): a 3D perspective + three clean
2D orthographic views — top-down (X-Y), front (X-Z), side (Y-Z). The 2D panels
give unambiguous axes (the old top-down-in-3D had an edge-on, unreadable Z axis).
World-frame columns chaser_wx/wy/wz + target_wx/wy/wz."""
import sys,csv
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
TCOL,CCOL='#1f6fe0','#d62728'
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
        return list(zip(*[(x,y,z) for x,y,z in zip(a,b,c) if None not in (x,y,z)]))
    cX,cY,cZ=clean(cx,cy,cz); tX,tY,tZ=clean(tx,ty,tz)
    fig=plt.figure(figsize=(13,10))
    # 1) 3D perspective
    ax=fig.add_subplot(2,2,1,projection='3d')
    ax.plot(tX,tY,tZ,color=TCOL,lw=1.8,label='target')
    ax.plot(cX,cY,cZ,color=CCOL,lw=1.8,label='chaser')
    ax.scatter([tX[0]],[tY[0]],[tZ[0]],c=TCOL,s=35); ax.scatter([cX[0]],[cY[0]],[cZ[0]],c=CCOL,s=35)
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
    ax.view_init(elev=22,azim=-60); ax.set_title('perspective (3D)',fontsize=11)
    ax.legend(loc='upper left',fontsize=9)
    # 2D orthographic panels: (subplot, xdata_t, ydata_t, xdata_c, ydata_c, xlabel, ylabel, title, equal)
    def panel(pos,tx_,ty_,cx_,cy_,xl,yl,ttl,equal=False):
        a=fig.add_subplot(2,2,pos)
        a.plot(tx_,ty_,color=TCOL,lw=1.8,label='target')
        a.plot(cx_,cy_,color=CCOL,lw=1.8,label='chaser')
        a.scatter([tx_[0]],[ty_[0]],c=TCOL,s=35); a.scatter([cx_[0]],[cy_[0]],c=CCOL,s=35)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3); a.set_title(ttl,fontsize=11)
        if equal: a.set_aspect('equal','datalim')
    panel(2,tX,tY,cX,cY,'X (m)','Y (m)','top-down / above (X-Y)',equal=True)
    panel(3,tX,tZ,cX,cZ,'X (m)','Z altitude (m)','front (X-Z)')
    panel(4,tY,tZ,cY,cZ,'Y (m)','Z altitude (m)','side (Y-Z)')
    fig.suptitle(title,fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.97]); plt.savefig(outp,dpi=120); print("wrote",outp)
main()
