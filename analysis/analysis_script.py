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