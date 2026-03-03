import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

BASE = os.path.expanduser('/Users/samreedhbhuyan/Desktop/Win_C/CERN/PRMON/data/analysis')
df = pd.read_csv(os.path.join(BASE, 'combined_dataset.csv'))
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

FEATURE_COLS = [
    'pss', 'rss', 'nthreads', 'nprocs',
    'utime', 'stime', 'rchar', 'wchar',
    'dpss_dt', 'cpu_eff', 'stime_ratio',
    'pss_per_proc', 'io_rate',
    'pss_roll_mean', 'pss_roll_std',
]

normal  = df[df['label'] == 0]
anomaly = df[df['label'] == 1]
norm_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='Normal')
anom_patch = mpatches.Patch(color='crimson',   alpha=0.7, label='Anomaly')

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
key_metrics = [
    ('pss',      'PSS (kB)',    'Memory (PSS) over Wall Time'),
    ('nthreads', 'nthreads',   'Thread Count over Wall Time'),
    ('cpu_eff',  'CPU Eff.',   'CPU Efficiency over Wall Time'),
]
for ax, (col, ylabel, title) in zip(axes, key_metrics):
    for run_id, grp in df.groupby('run_id'):
        is_anom = grp['label'].iloc[0] == 1
        ax.plot(grp['wtime'], grp[col],
                color='crimson' if is_anom else 'steelblue',
                alpha=0.85 if is_anom else 0.35,
                lw=2 if is_anom else 0.9)
    ax.set_xlabel('Wall Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(handles=[norm_patch, anom_patch])

plt.suptitle('Time Series: Normal vs Anomalous Runs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '01_timeseries_overview.png'), dpi=150)
plt.close()

key_feats = ['pss', 'nthreads', 'dpss_dt', 'cpu_eff',
             'stime_ratio', 'pss_per_proc', 'io_rate', 'pss_roll_std']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, feat in zip(axes.flatten(), key_feats):
    vals_n = normal[feat].clip(
        normal[feat].quantile(0.01),
        normal[feat].quantile(0.99)
    )
    vals_a = anomaly[feat].clip(
        anomaly[feat].quantile(0.01),
        anomaly[feat].quantile(0.99)
    )
    ax.hist(vals_n, bins=40, color='steelblue', alpha=0.6,
            density=True, label='Normal')
    ax.hist(vals_a, bins=40, color='crimson',   alpha=0.6,
            density=True, label='Anomaly')
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

plt.suptitle('Feature Distributions: Normal vs Anomaly', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '02_feature_distributions.png'), dpi=150)
plt.close()

pair_feats = ['pss', 'nthreads', 'cpu_eff', 'pss_per_proc', 'label']
pair_df = df[pair_feats].copy()
pair_df['label'] = pair_df['label'].map({0: 'Normal', 1: 'Anomaly'})
g = sns.pairplot(pair_df, hue='label',
                 palette={'Normal': 'steelblue', 'Anomaly': 'crimson'},
                 plot_kws={'alpha': 0.4, 's': 10},
                 diag_kind='kde')
g.figure.suptitle('Pairwise Feature Separability', y=1.02,
                   fontsize=12, fontweight='bold')
g.figure.savefig(os.path.join(FIG, '03_pairplot.png'), dpi=130, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(13, 10))
corr = df[FEATURE_COLS + ['label']].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, mask=mask, ax=ax, annot_kws={'size': 7},
            linewidths=0.4)
ax.set_title('Feature Correlation Matrix\n(lower triangle)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, '04_correlation_heatmap.png'), dpi=150)
plt.close()

print("EDA plots saved to:", FIG)

