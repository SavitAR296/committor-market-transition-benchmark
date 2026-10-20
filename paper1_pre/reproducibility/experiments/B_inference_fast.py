from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'tables_csf'

def delta_from_components(d,b,idx=None):
    if idx is None: return d.mean()/b.mean()
    return d[idx].mean()/b[idx].mean()

def mbb_indices(n,L,B,rng):
    k=(n+L-1)//L
    starts=rng.integers(0,max(1,n-L+1),size=(B,k))
    offs=np.arange(L)[None,None,:]
    idx=(starts[:,:,None]+offs).reshape(B,-1)[:,:n]
    return np.minimum(idx,n-1)

def stationary_indices(n,L,B,rng):
    out=np.empty((B,n),dtype=np.int32); p=1./L
    out[:,0]=rng.integers(n,size=B)
    for t in range(1,n):
        restart=rng.random(B)<p
        fresh=rng.integers(n,size=B)
        out[:,t]=np.where(restart,fresh,(out[:,t-1]+1)%n)
    return out

def ratios(d,b,idx):
    num=d[idx].mean(axis=1); den=b[idx].mean(axis=1)
    return num/den

def summarize(vals):
    vals=vals[np.isfinite(vals)]; p=min(1.,2*min(np.mean(vals<=0),np.mean(vals>=0)))
    return np.quantile(vals,[.025,.05,.95,.975]),p

def run(B=500,seed=20260915):
    F=pd.read_csv(OUT/'B_feature_attribution_forecasts.csv',parse_dates=['date'])
    F=F[F.headline.astype(bool)&F.y.notna()&F.p.notna()].copy()
    comps=[('pc2_logistic','ou_seqcal'),('pc2_gbt','ou_seqcal'),('raw_logistic','ou_seqcal')]
    rng=np.random.default_rng(seed); rows=[]
    for (s,h),g in F.groupby(['set','horizon']):
        meta=g[['date','y','clim','inB']].drop_duplicates('date').sort_values('date').set_index('date')
        P=g.pivot_table(index='date',columns='method',values='p',aggfunc='first').sort_index()
        common=meta.index.intersection(P.index); meta=meta.loc[common]; P=P.loc[common]
        for ma,mb in comps:
            ok=P[[ma,mb]].notna().all(1); m=meta.loc[ok]; pp=P.loc[ok]
            y=m.y.to_numpy(float); cl=m.clim.to_numpy(float); pa=pp[ma].to_numpy(float); pb=pp[mb].to_numpy(float); n=len(y)
            if n<100: continue
            d=(y-pb)**2-(y-pa)**2; base=(y-cl)**2; point=delta_from_components(d,base)
            for scheme,L in [('mbb',5),('mbb',10),('mbb',20),('mbb',40),('mbb',60),('stationary',10),('stationary',20),('stationary',40)]:
                idx=mbb_indices(n,L,B,rng) if scheme=='mbb' else stationary_indices(n,L,B,rng)
                vals=ratios(d,base,idx); q,p=summarize(vals)
                rows.append(dict(set=s,horizon=h,A=ma,B=mb,scheme=scheme,block=L,point=point,lo95=q[0],lo90=q[1],hi90=q[2],hi95=q[3],p_sign=p,n=n))
            # Segment bootstrap. headline excludes inB days, so use contiguous calendar gaps/operational episode boundaries by date gaps > 4d.
            dates=m.index.to_numpy(dtype='datetime64[D]'); gaps=np.r_[True,np.diff(dates).astype('timedelta64[D]').astype(int)>4]
            starts=np.where(gaps)[0]; cuts=np.r_[starts,n]; segs=[np.arange(cuts[i],cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]>cuts[i]]
            if len(segs)>=3:
                vals=np.empty(B,float)
                for j in range(B):
                    ix=[]
                    while len(ix)<n: ix.extend(segs[int(rng.integers(len(segs)))].tolist())
                    ii=np.asarray(ix[:n],int); vals[j]=delta_from_components(d,base,ii)
                q,p=summarize(vals)
                rows.append(dict(set=s,horizon=h,A=ma,B=mb,scheme='episode_segments',block=np.nan,point=point,lo95=q[0],lo90=q[1],hi90=q[2],hi95=q[3],p_sign=p,n=n))
    out=pd.DataFrame(rows); out.to_csv(OUT/'B_inference_robustness.csv',index=False)
    print(out.to_string(index=False)); return out
if __name__=='__main__': run()
