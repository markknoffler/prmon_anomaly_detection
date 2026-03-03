import pandas as pd
import numpy as np
import os
from sklearn.svm import OneClassSVM
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

# OCSVM is trained ONLY on normal data — this is the key difference from IF
# It defines a boundary around normal; anything outside is anomalous
X_normal = X_scaled[y == 0]

ocsvm = OneClassSVM(
    kernel='rbf',
    nu=0.1,       # upper bound on fraction of outliers in training data
    gamma='scale'
)
ocsvm.fit(X_normal)

raw_preds       = ocsvm.predict(X_scaled)   # +1 = normal, -1 = anomaly
decision_scores = -ocsvm.decision_function(X_scaled)  # higher = more anomalous

df['pred_ocsvm']  = (raw_preds == -1).astype(int)
df['score_ocsvm'] = decision_scores

print("="*50)
print("One-Class SVM (trained on normal data only)")
print("="*50)
print(classification_report(y, df['pred_ocsvm'], target_names=['Normal','Anomaly']))
print(f"ROC-AUC: {roc_auc_score(y, decision_scores):.3f}")

print("\nPer anomaly type detection rate:")
for atype, grp in df[df['label']==1].groupby('anomaly_type'):
    print(f"  {atype:20s}: {grp['pred_ocsvm'].mean()*100:.1f}% rows flagged")

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_title('One-Class SVM: Decision Score over Index\n'
             '(trained only on normal data — true production setup)',
             fontweight='bold')
for run_id, grp in df.groupby('run_id'):
    is_anom = grp['label'].iloc[0] == 1
    ax.plot(grp.index, grp['score_ocsvm'],
            color='crimson' if is_anom else 'steelblue',
            alpha=0.8 if is_anom else 0.3, lw=1)
ax.axhline(0, color='black', linestyle='--', lw=1, label='Decision boundary')
ax.set_ylabel('Decision Score (>0 = anomaly)')
ax.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', label='Normal'),
    plt.Line2D([0],[0], color='crimson',   label='Anomaly'),
    plt.Line2D([0],[0], color='black', linestyle='--', label='Boundary'),
])
plt.tight_layout()
plt.savefig(os.path.join(FIG, '06_one_class_svm.png'), dpi=150)
plt.close()

df[['run_id','wtime','pss','label','pred_ocsvm','score_ocsvm','anomaly_type']].to_csv(
    os.path.join(BASE, 'results_ocsvm.csv'), index=False
)
print("Results and plot saved.")

