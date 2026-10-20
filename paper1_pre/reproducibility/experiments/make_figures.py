from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures_csf'; OUT.mkdir(exist_ok=True)
T=ROOT/'tables_csf'

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close(fig)

def b_synthetic():
    d=pd.read_csv(T/'B_synthetic_phase.csv'); m=d.groupby(['tau_c','w','method'])[['AUC','BSS']].mean().reset_index();
    # tau_c read as float with inf supported
    taus=sorted(m.tau_c.unique(),key=lambda x:(np.isinf(x),x)); ws=[1,5,10,20,40,80]
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.0))
    for tau in taus:
        g=m[(m.tau_c==tau)&(m.method=='rolling_flex')].set_index('w').reindex(ws); o=m[(m.tau_c==tau)&(m.method=='oracle_pde')].set_index('w').reindex(ws)
        lab=r'$\tau_c=\infty$' if np.isinf(tau) else rf'$\tau_c={tau:g}$'
        ax[0].plot(ws,g.AUC,'o-',label=lab); ax[1].plot(ws,g.BSS,'o-',label=lab)
    oo=m[m.method=='oracle_pde'].groupby('w')[['AUC','BSS']].mean(); ax[0].plot(ws,oo.loc[ws].AUC,'k--',lw=1.5,label='oracle, avg.'); ax[1].plot(ws,oo.loc[ws].BSS,'k--',lw=1.5,label='oracle, avg.')
    for a,y,t in [(ax[0],'AUC','(a) discrimination'),(ax[1],'Brier skill','(b) probability skill')]: a.set_xscale('log',base=2); a.set_xticks(ws,ws); a.set_xlabel('rolling width $w$'); a.set_ylabel(y); a.set_title(t)
    ax[0].legend(frameon=False,ncol=2,fontsize=8); save(fig,'B_fig2_synthetic_phase')

def b_feature():
    d=pd.read_csv(T/'B_feature_attribution_metrics.csv'); methods=['ou_raw','ou_seqcal','pc2_logistic','pc2_gbt','raw_logistic','raw_gbt']; labs=['OU raw','OU seq-cal','PC2 logit','PC2 GBT','raw logit','raw GBT']; sets=['full11','no_option9','common7']; sl=['full 11','no options (9)','common 7']
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,h in enumerate([5,20]):
        for col,metric in enumerate(['AUC','BSS']):
            ax=axs[row,col]; x=np.arange(len(sets)); width=.12
            for k,(meth,lab) in enumerate(zip(methods,labs)):
                vals=[]
                for s in sets:
                    q=d[(d.set==s)&(d.horizon==h)&(d.method==meth)]; vals.append(q[metric].iloc[0] if len(q) else np.nan)
                ax.bar(x+(k-2.5)*width,vals,width=width,label=lab)
            ax.set_xticks(x,sl); ax.set_ylabel(metric); ax.set_title(f'{h}-day horizon: {metric}')
            if metric=='BSS': ax.axhline(0,color='k',lw=.7)
    axs[0,0].legend(frameon=False,ncol=3,fontsize=7); save(fig,'B_fig3_feature_attribution')

def b_robustness():
    inf=pd.read_csv(T/'B_inference_robustness.csv'); tar=pd.read_csv(T/'B_target_sensitivity.csv')
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.0))
    # focus common7 h20: matched baselines vs sequential OU
    q=inf[(inf['set']=='common7')&(inf.horizon==20)&(inf.scheme.isin(['mbb','stationary']))]
    methods=['pc2_logistic','pc2_gbt','raw_logistic']; labels=['PC2 logistic','PC2 GBT','raw logistic']
    for j,(m,l) in enumerate(zip(methods,labels)):
        g=q[q.A==m].copy(); # x code separate schemes
        # moving block only for clean plot
        g=g[g.scheme=='mbb'].sort_values('block'); x=g.block.to_numpy(float)
        ax[0].errorbar(x+j*.8,g.point,yerr=[g.point-g.lo95,g.hi95-g.point],fmt='o-',capsize=3,label=l)
    ax[0].axhline(0,color='k',lw=.7); ax[0].set_xlabel('moving-block length (days)'); ax[0].set_ylabel(r'$\Delta$BSS vs sequential OU'); ax[0].set_title('(a) dependence robustness, common 7'); ax[0].legend(frameon=False,fontsize=8)
    # target sensitivity average annual AUC
    s=tar.groupby(['target_q','horizon']).AUC.agg(['mean','std']).reset_index()
    for h,mark in [(5,'o'),(20,'s')]:
        g=s[s.horizon==h]; ax[1].errorbar(g.target_q,g['mean'],yerr=g['std'],marker=mark,capsize=3,label=f'{h}-day')
    ax[1].set_xlabel('stress-set training quantile'); ax[1].set_ylabel('mean annual AUC'); ax[1].set_title('(b) target-set sensitivity'); ax[1].legend(frameon=False)
    save(fig,'B_fig4_robustness')

if __name__=='__main__':
    b_synthetic(); b_feature(); b_robustness(); print('B figures regenerated')
