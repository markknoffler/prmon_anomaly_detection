import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

BASE = os.path.expanduser('/Users/samreedhbhuyan/Desktop/Win_C/CERN/PERMON/data/analysis')
FIG  = os.path.join(BASE, 'figures')
df   = pd.read_csv(os.path.join(BASE, 'combined_dataset.csv'))

FEATURE_COLS = [
    'pss', 'rss', 'nthreads', 'nprocs',
    'utime', 'stime', 'rchar', 'wchar',
    'dpss_dt', 'cpu_eff', 'stime_ratio',
    'pss_per_proc', 'io_rate',
    'pss_roll_mean', 'pss_roll_std',
]

X = df[FEATURE_COLS].values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# contamination = fraction of anomalous rows in dataset
# 4 anomaly runs / 10 total runs ≈ 0.4 but rows per run vary,
# use 0.25 as conservative estimate
iso = IsolationForest(
    n_estimators=300,
    contamination=0.25,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
iso.fit(X_scaled)

raw_preds  = iso.predict(X_scaled)         # +1 = normal, -1 = anomaly
scores     = -iso.score_samples(X_scaled)  # higher = more anomalous

df['pred_iso']   = (raw_preds == -1).astype(int)
df['score_iso']  = scores

print("="*50)
print("Isolation Forest")
print("="*50)
print(classification_report(y, df['pred_iso'], target_names=['Normal','Anomaly']))
print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}")

print("\nPer anomaly type detection rate:")
for atype, grp in df[df['label']==1].groupby('anomaly_type'):
    print(f"  {atype:20s}: {grp['pred_iso'].mean()*100:.1f}% rows flagged")

# ── Anomaly score plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].set_title('Isolation Forest: Anomaly Score over Time', fontweight='bold')
for run_id, grp in df.groupby('run_id'):
    is_anom = grp['label'].iloc[0] == 1
    axes[0].plot(grp.index, grp['score_iso'],
                 color='crimson' if is_anom else 'steelblue',
                 alpha=0.8 if is_anom else 0.3, lw=1)
axes[0].set_ylabel('Anomaly Score (higher=more anomalous)')
axes[0].legend(handles=[
    plt.Line2D([0],[0], color='steelblue', label='Normal'),
    plt.Line2D([0],[0], color='crimson',   label='Anomaly')
])

axes[1].set_title('Isolation Forest: PSS with Flagged Points', fontweight='bold')
axes[1].plot(df.index, df['pss'], color='steelblue', alpha=0.4, lw=0.7, label='PSS')
tp = df[(df['label']==1) & (df['pred_iso']==1)]
fp = df[(df['label']==0) & (df['pred_iso']==1)]
fn = df[(df['label']==1) & (df['pred_iso']==0)]
axes[1].scatter(tp.index, tp['pss'], color='crimson', s=15,
                zorder=5, label=f'True Positive ({len(tp)})')
axes[1].scatter(fp.index, fp['pss'], color='orange', s=15, marker='x',
                zorder=5, label=f'False Positive ({len(fp)})')
axes[1].scatter(fn.index, fn['pss'], color='purple', s=15, marker='^',
                zorder=5, label=f'False Negative ({len(fn)})')
axes[1].set_ylabel('PSS (kB)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG, '05_isolation_forest.png'), dpi=150)
plt.close()

df[['run_id','wtime','pss','label','pred_iso','score_iso','anomaly_type']].to_csv(
    os.path.join(BASE, 'results_isolation_forest.csv'), index=False
)
print("Results and plot saved.")

