# analysis_script.py
"""
Saves outputs (plots and summaries) to ../outputs/ when run from analysis/ folder.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import zipfile

# ------------------
# Paths & output dir
# ------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


D1_PATH = os.path.join(DATA_DIR, 'dataset1.csv')
D2_PATH = os.path.join(DATA_DIR, 'dataset2.csv')
ZIP_OUT = os.path.join(BASE_DIR, 'assessment2_submission_bundle.zip')

# ------------------
# Helper functions
# ------------------

def try_parse_datetime(series):
    try:
        return pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')
    
def cliffs_delta(x, y):
    # returns Cliff's delta (x > y positive)
    nx = len(x); ny = len(y)
    more = 0
    less = 0
    for xi in x:
        more += np.sum(y < xi)
        less += np.sum(y > xi)
    delta = (more - less) / (nx * ny)
    return delta

# ------------------
# Load data
# ------------------
print('Loading datasets...')
d1 = pd.read_csv(D1_PATH)
d2 = pd.read_csv(D2_PATH)
print('dataset1 rows:', len(d1))
print('dataset2 rows:', len(d2))

# ------------------
# Clean & parse
# ------------------
# Parse likely datetime columns in dataset1
for col in ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']:
    if col in d1.columns:
        d1[col + '_dt'] = try_parse_datetime(d1[col])


if 'time' in d2.columns:
    d2['time_dt'] = try_parse_datetime(d2['time'])

# Numeric coercion (safe)
for col in ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset', 'risk']:
    if col in d1.columns:
        d1[col] = pd.to_numeric(d1[col], errors='coerce')


for col in ['bat_landing_number','food_availability','rat_minutes','rat_arrival_number','hours_after_sunset']:
    if col in d2.columns:
        d2[col] = pd.to_numeric(d2[col], errors='coerce')

# Categorise seconds_after_rat_arrival into bins (helpful for plotting)
if 'seconds_after_rat_arrival' in d1.columns:
    d1['seconds_after_rat_arrival_cat'] = pd.cut(
        d1['seconds_after_rat_arrival'],
        bins=[-1, 0, 10, 60, 300, 1e9],
        labels=['before_or_at_arrival', '0-10s', '10-60s', '1-5min', '>5min']
    )

# Save cleaned head files for reproducibility
d1.head(200).to_csv(os.path.join(OUT_DIR, 'dataset1_head.csv'), index=False)
d2.head(200).to_csv(os.path.join(OUT_DIR, 'dataset2_head.csv'), index=False)

# ------------------
# Exploratory plots
# ------------------
# 1) Histogram of bat_landing_to_food
if 'bat_landing_to_food' in d1.columns:
    plt.figure()
    plt.hist(d1['bat_landing_to_food'].dropna(), bins=30)
    plt.title('Distribution of bat_landing_to_food (seconds)')
    plt.xlabel('Seconds until approach food')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hist_bat_landing_to_food.png'))
    plt.close()

# 2) Boxplot by risk
if 'bat_landing_to_food' in d1.columns and 'risk' in d1.columns:
    plt.figure()
    data0 = d1.loc[d1['risk'] == 0, 'bat_landing_to_food'].dropna()
    data1 = d1.loc[d1['risk'] == 1, 'bat_landing_to_food'].dropna()
    plt.boxplot([data0, data1], labels=['risk=0', 'risk=1'])
    plt.title('bat_landing_to_food by risk group')
    plt.ylabel('Seconds until approach food')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'box_bat_landing_by_risk.png'))
    plt.close()


# 3) Scatter from dataset2: rat arrivals vs bat landings
if 'rat_arrival_number' in d2.columns and 'bat_landing_number' in d2.columns:
    plt.figure()
    plt.scatter(d2['rat_arrival_number'], d2['bat_landing_number'])
    plt.title('Rat arrivals vs Bat landings (30-min periods)')
    plt.xlabel('rat_arrival_number (per 30 min)')
    plt.ylabel('bat_landing_number (per 30 min)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scatter_rats_vs_bats.png'))
    plt.close()


# ------------------
# Inferential analyses
# ------------------
# Prepare df for modelling: keep rows with required vars
model_df = d1.copy()

# Mann-Whitney U (compare bat_landing_to_food by risk label)
mw_result = None
if 'bat_landing_to_food' in model_df.columns and 'risk' in model_df.columns:
    grp0 = model_df.loc[model_df['risk'] == 0, 'bat_landing_to_food'].dropna()
    grp1 = model_df.loc[model_df['risk'] == 1, 'bat_landing_to_food'].dropna()
    if len(grp0) > 0 and len(grp1) > 0:
        u_stat, p_val = mannwhitneyu(grp0, grp1, alternative='two-sided')
        n0, n1 = len(grp0), len(grp1)
        mu_U = n0 * n1 / 2.0
        sigma_U = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12.0)
        z = (u_stat - mu_U) / sigma_U
        r = z / np.sqrt(n0 + n1)
        delta = cliffs_delta(grp1.values, grp0.values)
        mw_result = {
            'u_stat': int(u_stat),
            'pvalue': float(p_val),
            'z': float(z),
            'r': float(r),
            'n0': n0,
            'n1': n1,
            'cliffs_delta': float(delta)
        }
        with open(os.path.join(OUT_DIR, 'mannwhitneyu_result.txt'), 'w') as f:
            f.write(str(mw_result))