import os,sys,time,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_daily import crisis_labels
from committor import Grid2D
OUT=ROOT/'tables_csf'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures_csf'; FIG.mkdir(exist_ok=True)

FULL=['log_rv20','log_rv5','log_act_rel','dd20','ret20','one_minus_n','n_sd','ews_var','ews_ar1','log_vix','vrp']
NO_OPTION=[c for c in FULL if c not in ['log_vix','vrp']]
COMMON=[c for c in NO_OPTION if c not in ['one_minus_n','n_sd']]
SETS={'full11':FULL,'no_option9':NO_OPTION,'common7':COMMON}

class OUModel:
    def __init__(self,z,dt=1.0):
        z0=z[:-1]; dz=z[1:]-z[:-1]
        X=np.c_[z0,np.ones(len(z0))]
        B,*_=np.linalg.lstsq(X,dz/dt,rcond=None)
        self.A=B[:2].T; self.b=B[2]
        resid=dz/dt-X@B
        self.D=np.cov(resid.T)*dt + 1e-6*np.eye(2)
    def fields(self,pts):
        return pts@self.A.T+self.b, np.repeat(self.D[None],len(pts),axis=0)

def interp_grid(G,vals,pts):
    r=RegularGridInterpolator((G.x,G.y),vals.reshape(G.nx,G.ny),bounds_error=False,fill_value=None)
    lo=np.array([G.x[0],G.y[0]]); hi=np.array([G.x[-1],G.y[-1]])
    return np.clip(r(np.clip(pts,lo,hi)),0,1)

def grid_from_train(z,n=40):
    lo=np.quantile(z,0.005,axis=0); hi=np.quantile(z,0.995,axis=0)
    span=np.maximum(hi-lo,1.0)
    lo=lo-0.18*span; hi=hi+0.18*span
    return Grid2D(lo[0],hi[0],lo[1],hi[1],n,n)

def fit_ou_scores(ztr,zte,lrv_tr,horizons=(5,20),grid_n=40):
    G=grid_from_train(ztr,grid_n)
    ou=OUModel(ztr); f,D=ou.fields(G.pts); L=G.generator(f,D)
    cors=np.array([np.corrcoef(ztr[:,j],lrv_tr)[0,1] for j in range(2)])
    if not np.all(np.isfinite(cors)) or np.linalg.norm(cors)<1e-9: cors=np.array([1.,0.])
    w=cors/np.linalg.norm(cors)
    s_tr=ztr@w; s_grid=G.pts@w
    thr=np.quantile(s_tr,0.90)
    maskB=s_grid>=thr
    path=G.hitting_prob_path(L,maskB,max(horizons),dt=1.0)
    out={}
    for h in horizons:
        out[h]=(interp_grid(G,path[int(h)],ztr),interp_grid(G,path[int(h)],zte))
    return out,w,thr

def safe_prob_logistic(Xtr,ytr,Xte,C=1.0):
    if np.sum(ytr==1)<3 or np.sum(ytr==0)<3:
        return np.full(len(Xte),np.mean(ytr))
    m=LogisticRegression(C=C,max_iter=3000,class_weight=None,random_state=0).fit(Xtr,ytr)
    return m.predict_proba(Xte)[:,1]

def safe_gbt(Xtr,ytr,Xte):
    if np.sum(ytr==1)<3 or np.sum(ytr==0)<3:
        return np.full(len(Xte),np.mean(ytr))
    m=GradientBoostingClassifier(n_estimators=120,learning_rate=0.03,max_depth=2,min_samples_leaf=20,subsample=0.9,random_state=0)
    m.fit(Xtr,ytr)
    return m.predict_proba(Xte)[:,1]

def sequential_recal(prior_s,prior_y,s_te,clim):
    if len(prior_y)<100 or np.sum(np.asarray(prior_y)==1)<5 or np.sum(np.asarray(prior_y)==0)<20:
        return np.full(len(s_te),clim)
    s=np.asarray(prior_s,float); y=np.asarray(prior_y,int)
    if np.nanstd(s)<1e-12: return np.full(len(s_te),np.mean(y))
    lr=LogisticRegression(C=1.0,max_iter=2000).fit(np.nan_to_num(s,nan=np.nanmedian(s))[:,None],y)
    return lr.predict_proba(np.nan_to_num(s_te,nan=np.nanmedian(s))[:,None])[:,1]

def feature_attribution():
    F=pd.read_csv(ROOT/'tables'/'features_SPX.csv',parse_dates=['date'])
    # Matched complete-case sample across all feature sets.
    F=F.dropna(subset=FULL).reset_index(drop=True)
    labels={h:crisis_labels(F,h,rule='rv_dd')[0] for h in [5,20]}
    _,inB,_=crisis_labels(F,20,rule='rv_dd'); F['inB']=inB
    years=[y for y in sorted(F.date.dt.year.unique()) if y>=2008]
    fp=OUT/'B_feature_attribution_forecasts.csv'; lp=OUT/'B_pca_loadings.csv'
    if fp.exists():
        prev=pd.read_csv(fp,parse_dates=['date']); rows=prev.to_dict('records'); done_years=set(prev.year.unique())
    else:
        prev=pd.DataFrame(); rows=[]; done_years=set()
    loading_rows=pd.read_csv(lp).to_dict('records') if lp.exists() else []
    # strict sequential calibration history reconstructed from prior OOS raw scores.
    hist={(ss,h):([],[]) for ss in SETS for h in [5,20]}
    if len(prev):
        q=prev[(prev.method=='ou_raw') & prev.headline.astype(bool) & prev.y.notna() & prev.p.notna()]
        for (ss,h),g in q.groupby(['set','horizon']):
            hist[(ss,int(h))]=(g.p.astype(float).tolist(),g.y.astype(int).tolist())
    for Y in years:
        if Y in done_years:
            print('feature attribution skip',Y,flush=True); continue
        trmask=F.date<pd.Timestamp(f'{Y}-01-01'); temask=(F.date>=pd.Timestamp(f'{Y}-01-01'))&(F.date<pd.Timestamp(f'{Y+1}-01-01'))
        tr=F[trmask]; te=F[temask]
        if len(te)==0 or len(tr)<600: continue
        tridx=np.where(trmask)[0]; teidx=np.where(temask)[0]
        for sname,cols in SETS.items():
            scaler=StandardScaler().fit(tr[cols])
            Xtr=scaler.transform(tr[cols]); Xte=scaler.transform(te[cols])
            pca=PCA(n_components=2,random_state=0).fit(Xtr)
            Ztr=pca.transform(Xtr); Zte=pca.transform(Xte)
            # standardize embedded coords using train only
            zsc=StandardScaler().fit(Ztr); Ztr=zsc.transform(Ztr); Zte=zsc.transform(Zte)
            for j in range(2):
                for k,c in enumerate(cols):
                    loading_rows.append(dict(year=Y,set=sname,pc=j+1,feature=c,loading=pca.components_[j,k],explained=pca.explained_variance_ratio_[j]))
            ouscores,_,_=fit_ou_scores(Ztr,Zte,tr['log_rv20'].values)
            for h in [5,20]:
                yall=labels[h]; ytr=yall[tridx].copy(); yte=yall[teidx].copy()
                # prevent training labels whose horizon crosses the year boundary from being used
                oktr=np.isfinite(ytr)
                if h>0 and len(oktr)>=h: oktr[-h:]=False
                clim=float(np.nanmean(ytr[oktr])) if np.any(oktr) else 0.01
                # Current day in operational target set excluded from headline scoring.
                score_ok=np.isfinite(yte) & (~te['inB'].values.astype(bool))
                methods={}
                methods['clim']=np.full(len(te),clim)
                methods['pc2_logistic']=safe_prob_logistic(Ztr[oktr],ytr[oktr].astype(int),Zte,C=1.0)
                methods['pc2_gbt']=safe_gbt(Ztr[oktr],ytr[oktr].astype(int),Zte)
                methods['raw_logistic']=safe_prob_logistic(Xtr[oktr],ytr[oktr].astype(int),Xte,C=0.2)
                methods['raw_gbt']=safe_gbt(Xtr[oktr],ytr[oktr].astype(int),Xte)
                rawtr,rawte=ouscores[h]
                methods['ou_raw']=rawte
                ps,py=hist[(sname,h)]
                methods['ou_seqcal']=sequential_recal(ps,py,rawte,clim)
                # append current OOS raw scores only after calibration was generated
                for ss,yy,valid in zip(rawte,yte,score_ok):
                    if valid and np.isfinite(ss): ps.append(float(ss)); py.append(int(yy))
                hist[(sname,h)]=(ps,py)
                for method,p in methods.items():
                    for d,pp,yy,valid,ib in zip(te.date,p,yte,score_ok,te.inB.values):
                        rows.append(dict(date=d,year=Y,set=sname,horizon=h,method=method,p=float(pp),y=yy,headline=bool(valid),inB=bool(ib),clim=clim))
        pd.DataFrame(rows).to_csv(OUT/'B_feature_attribution_forecasts.csv',index=False)
        pd.DataFrame(loading_rows).to_csv(OUT/'B_pca_loadings.csv',index=False)
        print('feature attribution year',Y,'done',flush=True)
    df=pd.DataFrame(rows)
    metrics=[]
    for (s,h,m),g in df.groupby(['set','horizon','method']):
        g=g[g.headline & np.isfinite(g.y) & np.isfinite(g.p)]
        if len(g)<50 or g.y.sum()<3: continue
        auc=roc_auc_score(g.y,g.p)
        bs=brier_score_loss(g.y,g.p)
        # OOS year-specific climatology forecast is stored rowwise
        bs0=np.mean((g.y-g.clim)**2)
        metrics.append(dict(set=s,horizon=h,method=m,AUC=auc,Brier=bs,BSS=1-bs/bs0,n=len(g),events=int(g.y.sum())))
    md=pd.DataFrame(metrics); md.to_csv(OUT/'B_feature_attribution_metrics.csv',index=False)
    return df,md

# ---------- Synthetic observation-design phase diagram ----------
def simulate_doublewell(T=14000,dt=0.05,tau_c=20.0,sigma_x=0.72,sigma_c=0.18,seed=0):
    rng=np.random.default_rng(seed); x=np.empty(T); c=np.empty(T); x[0]=-1.0; c[0]=0.0
    for i in range(T-1):
        if np.isinf(tau_c):
            c[i+1]=0.0
        else:
            c[i+1]=c[i]+(-c[i]/tau_c)*dt+sigma_c*np.sqrt(dt)*rng.normal()
            c[i+1]=np.clip(c[i+1],-0.45,0.45)
        drift=-(x[i]**3-x[i]-c[i])
        x[i+1]=x[i]+drift*dt+sigma_x*np.sqrt(dt)*rng.normal()
        x[i+1]=np.clip(x[i+1],-2.2,2.2)
    return x,c

def future_hit_labels(x,h_steps,target=0.65):
    hit=x>=target; n=len(x); y=np.zeros(n,float); nxt=np.where(hit)[0]
    for t in range(n-h_steps):
        j=np.searchsorted(nxt,t+1)
        y[t]=1.0 if j<len(nxt) and nxt[j]<=t+h_steps else 0.0
    y[n-h_steps:]=np.nan
    return y

def rolling_state(x,c,w):
    if w==1: return np.c_[x,c]
    sx=pd.Series(x).rolling(w,min_periods=w).mean().values
    sc=pd.Series(c).rolling(w,min_periods=w).mean().values
    return np.c_[sx,sc]

def true_oracle_grid(tau_c,sigma_x=0.72,sigma_c=0.18,h=2.0,target=0.65):
    # True generator of the latent nonlinear SDE; finite-horizon PDE defines the oracle probability.
    G=Grid2D(-2.2,2.2,-0.5,0.5,55,35)
    x=G.pts[:,0]; c=G.pts[:,1]
    fx=-(x**3-x-c)
    if np.isinf(tau_c):
        fc=np.zeros_like(c); sc=1e-5
    else:
        fc=-c/tau_c; sc=sigma_c
    f=np.c_[fx,fc]
    D=np.zeros((len(x),2,2)); D[:,0,0]=sigma_x**2; D[:,1,1]=sc**2
    L=G.generator(f,D); maskB=x>=target
    v=G.hitting_prob(L,maskB,h,dt=0.05)
    return G,v

def synthetic_phase_diagram(reps=8,seed=20260914):
    fp=OUT/'B_synthetic_phase.csv'
    prev=pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    rows=prev.to_dict('records') if len(prev) else []
    done=set()
    if len(prev):
        for (tt,rr),g in prev.groupby(['tau_c','rep']):
            if g['w'].nunique()>=6:
                done.add((str(tt),int(rr)))
    rng=np.random.default_rng(seed); dt=0.05; h_steps=40; target=0.65
    for tau in [5.,20.,80.,np.inf]:
        Gtrue,vtrue=true_oracle_grid(tau,h=h_steps*dt,target=target)
        taukey='inf' if np.isinf(tau) else str(float(tau))
        for rep in range(reps):
            simseed=int(rng.integers(1,2**31-1))
            if (taukey,rep) in done:
                print('synthetic skip',tau,rep,flush=True); continue
            x,c=simulate_doublewell(tau_c=tau,seed=simseed)
            y=future_hit_labels(x,h_steps,target=target)
            split=int(len(x)*0.60); base=np.arange(len(x)); latent=np.c_[x,c]
            for w in [1,5,10,20,40,80]:
                Z=rolling_state(x,c,w); valid=np.isfinite(Z).all(1)&np.isfinite(y)
                tr=valid&(base<split); te=valid&(base>=split)&(x<target)
                if y[te].sum()<3: continue
                oracle=interp_grid(Gtrue,vtrue,latent[te])
                latent_flex=safe_gbt(latent[tr],y[tr].astype(int),latent[te])
                flex=safe_gbt(Z[tr],y[tr].astype(int),Z[te])
                sc=StandardScaler().fit(Z[tr]); ztr=sc.transform(Z[tr]); zte=sc.transform(Z[te])
                try:
                    # OU is fitted per observation step; 40 steps correspond to the same physical horizon h=2.
                    ous,_,_=fit_ou_scores(ztr,zte,Z[tr][:,0],horizons=(h_steps,),grid_n=32)
                    ou=ous[h_steps][1]
                except Exception:
                    ou=np.full(np.sum(te),np.nan)
                yy=y[te]; clim=np.mean(y[tr])
                for method,pred in [('oracle_pde',oracle),('latent_flex',latent_flex),('rolling_flex',flex),('rolling_ou',ou)]:
                    ok=np.isfinite(pred)
                    if ok.sum()<50 or yy[ok].sum()<3: continue
                    auc=roc_auc_score(yy[ok],pred[ok]); bs=brier_score_loss(yy[ok],pred[ok]); bs0=np.mean((yy[ok]-clim)**2)
                    rows.append(dict(tau_c=('inf' if np.isinf(tau) else tau),rep=rep,w=w,method=method,AUC=auc,BSS=1-bs/bs0,n=ok.sum(),events=int(yy[ok].sum())))
            print('synthetic tau',tau,'rep',rep,'done',flush=True)
            pd.DataFrame(rows).to_csv(OUT/'B_synthetic_phase.csv',index=False)
    return pd.DataFrame(rows)


# NOTE: functions below are also callable via a separate helper command from Python.
def _delta_bss(g,ma,mb,idx):
    a=g[g.method==ma].set_index('date'); b=g[g.method==mb].set_index('date')
    dates=np.array(sorted(set(a.index)&set(b.index)))
    if len(dates)==0: return np.nan
    # idx indexes into dates and may contain repeats.
    ds=dates[np.asarray(idx,int)]
    ya=a.loc[ds,'y'].to_numpy(float); pa=a.loc[ds,'p'].to_numpy(float); pb=b.loc[ds,'p'].to_numpy(float); cl=a.loc[ds,'clim'].to_numpy(float)
    base=np.mean((ya-cl)**2)
    return (1-np.mean((ya-pa)**2)/base)-(1-np.mean((ya-pb)**2)/base)

def _moving_blocks(n,L,rng):
    out=[]
    while len(out)<n:
        s=int(rng.integers(0,max(1,n-L+1))); out.extend(range(s,min(n,s+L)))
    return np.array(out[:n])

def _stationary_blocks(n,L,rng):
    p=1.0/L; out=[]; cur=int(rng.integers(n))
    while len(out)<n:
        out.append(cur)
        if rng.random()<p: cur=int(rng.integers(n))
        else: cur=(cur+1)%n
    return np.array(out)

def inference_robustness(B=250,seed=20260915):
    F=pd.read_csv(OUT/'B_feature_attribution_forecasts.csv',parse_dates=['date'])
    F=F[F.headline.astype(bool)&F.y.notna()&F.p.notna()]
    comps=[('pc2_logistic','ou_seqcal'),('pc2_gbt','ou_seqcal'),('raw_logistic','ou_seqcal')]
    rows=[]; rng=np.random.default_rng(seed)
    for (ss,h),g0 in F.groupby(['set','horizon']):
        wide=g0.pivot_table(index='date',columns='method',values='p')
        meta=g0.groupby('date').agg(y=('y','first'),clim=('clim','first'),inB=('inB','first')).reindex(wide.index)
        for ma,mb in comps:
            if ma not in wide or mb not in wide: continue
            ok=np.isfinite(wide[ma].values)&np.isfinite(wide[mb].values)&np.isfinite(meta.y.values)
            ya=meta.y.values[ok].astype(float); pa=wide[ma].values[ok].astype(float); pb=wide[mb].values[ok].astype(float); cl=meta.clim.values[ok].astype(float); inb=meta.inB.values[ok].astype(bool)
            n=len(ya)
            if n<100: continue
            def db(idx):
                yy=ya[idx]; aa=pa[idx]; bb=pb[idx]; cc=cl[idx]; base=np.mean((yy-cc)**2)
                return np.mean((yy-bb)**2)/base-np.mean((yy-aa)**2)/base
            point=db(np.arange(n))
            for scheme,L in [('mbb',5),('mbb',10),('mbb',20),('mbb',40),('mbb',60),('stationary',10),('stationary',20),('stationary',40)]:
                vals=np.empty(B)
                for bi in range(B):
                    idx=_moving_blocks(n,L,rng) if scheme=='mbb' else _stationary_blocks(n,L,rng)
                    vals[bi]=db(idx)
                psg=2*min(np.mean(vals<=0),np.mean(vals>=0))
                rows.append(dict(set=ss,horizon=h,A=ma,B=mb,scheme=scheme,block=L,point=point,lo90=np.quantile(vals,.05),hi90=np.quantile(vals,.95),lo95=np.quantile(vals,.025),hi95=np.quantile(vals,.975),p_sign=min(1,psg)))
            starts=np.where(inb & ~np.r_[False,inb[:-1]])[0]; cuts=np.unique(np.r_[0,starts,n]); segs=[np.arange(cuts[i],cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]>cuts[i]]
            if len(segs)>=3:
                vals=np.empty(B)
                for bi in range(B):
                    idx=[]
                    while len(idx)<n: idx.extend(segs[int(rng.integers(len(segs)))].tolist())
                    vals[bi]=db(np.asarray(idx[:n]))
                psg=2*min(np.mean(vals<=0),np.mean(vals>=0))
                rows.append(dict(set=ss,horizon=h,A=ma,B=mb,scheme='episode_segments',block=np.nan,point=point,lo90=np.quantile(vals,.05),hi90=np.quantile(vals,.95),lo95=np.quantile(vals,.025),hi95=np.quantile(vals,.975),p_sign=min(1,psg)))
    out=pd.DataFrame(rows); out.to_csv(OUT/'B_inference_robustness.csv',index=False); return out

def target_sensitivity(grid_n=36):
    F=pd.read_csv(ROOT/'tables'/'features_SPX.csv',parse_dates=['date']).dropna(subset=FULL).reset_index(drop=True)
    labels={h:crisis_labels(F,h,rule='rv_dd')[0] for h in [5,20]}; _,inB,_=crisis_labels(F,20,rule='rv_dd'); F['inB']=inB
    years=[y for y in sorted(F.date.dt.year.unique()) if y>=2008]
    rows=[]
    cols=COMMON
    for Y in years:
        trmask=F.date<pd.Timestamp(f'{Y}-01-01'); temask=(F.date>=pd.Timestamp(f'{Y}-01-01'))&(F.date<pd.Timestamp(f'{Y+1}-01-01'))
        tr=F[trmask]; te=F[temask]
        if len(te)==0: continue
        sc=StandardScaler().fit(tr[cols]); Xtr=sc.transform(tr[cols]); Xte=sc.transform(te[cols]); pca=PCA(2).fit(Xtr)
        Ztr=pca.transform(Xtr); Zte=pca.transform(Xte); zsc=StandardScaler().fit(Ztr); Ztr=zsc.transform(Ztr); Zte=zsc.transform(Zte)
        G=grid_from_train(Ztr,grid_n); ou=OUModel(Ztr); f,D=ou.fields(G.pts); L=G.generator(f,D)
        cors=np.array([np.corrcoef(Ztr[:,j],tr.log_rv20.values)[0,1] for j in range(2)]); w=cors/np.linalg.norm(cors)
        strn=Ztr@w; sg=G.pts@w
        for q in [0.85,0.90,0.95]:
            maskB=sg>=np.quantile(strn,q); path=G.hitting_prob_path(L,maskB,20,dt=1.0)
            for h in [5,20]:
                p=interp_grid(G,path[h],Zte); y=labels[h][np.where(temask)[0]]; ok=np.isfinite(y)&(~te.inB.values.astype(bool))
                if ok.sum()<50 or y[ok].sum()<3: continue
                rows.append(dict(year=Y,target_q=q,horizon=h,AUC=roc_auc_score(y[ok],p[ok]),n=ok.sum(),events=int(y[ok].sum())))
        print('target sens year',Y,flush=True)
    out=pd.DataFrame(rows); out.to_csv(OUT/'B_target_sensitivity.csv',index=False); return out


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('task', choices=['feature','synthetic','target','all'])
    ap.add_argument('--reps', type=int, default=8)
    a=ap.parse_args(); st=time.time()
    if a.task in ('feature','all'): feature_attribution()
    if a.task in ('synthetic','all'): synthetic_phase_diagram(reps=a.reps)
    if a.task in ('target','all'): target_sensitivity()
    print('done sec',time.time()-st)
