import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, roc_auc_score

BASE = os.path.expanduser('/Users/samreedhbhuyan/Desktop/Win_C/CERN/PERMON/data/analysis')
df = pd.read_csv(os.path.join(BASE, 'combined_dataset.csv'))

Z_THRESHOLD = 3.0
df['pred_zscore'] = (np.abs(df['pss_zscore']) > Z_THRESHOLD).astype(int)

y_true = df['label'].values
y_pred = df['pred_zscore'].values

print("="*50)
print("Rolling Z-Score Detector (PSS, threshold=3σ)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Normal','Anomaly']))
print(f"ROC-AUC: {roc_auc_score(y_true, np.abs(df['pss_zscore'])):.3f}")

# Per anomaly type breakdown
print("\nPer anomaly type detection rate:")
for atype, grp in df[df['label']==1].groupby('anomaly_type'):
    detected = grp['pred_zscore'].mean()
    print(f"  {atype:20s}: {detected*100:.1f}% rows flagged")

df[['run_id','wtime','pss','pss_zscore','label','pred_zscore','anomaly_type']].to_csv(
    os.path.join(BASE, 'results_zscore.csv'), index=False
)
print(f"\nResults saved.")

