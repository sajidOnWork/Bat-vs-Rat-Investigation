#!/usr/bin/env python3
"""
project_analysis_objective3.py
HIT140 – Foundations of Data Science (Assessment 3 / Objective 2)

Investigations:
 A.  Do bats perceive rats as predators?
 B.  Do these behaviours change across seasons?

Required files (in same folder):
    dataset1.csv
    dataset2.csv
Outputs:
    project_outputs_obj3/  ->  charts, CSVs, and summary text
"""

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols
import pickle




# --------------------------------------------------------------------
# Paths and folders
# --------------------------------------------------------------------
DATA_DIR = Path("data")
OUT = ".." / Path("project_outputs_obj3")
OUT.mkdir(exist_ok=True)

d1_path = ".." / DATA_DIR / "dataset1.csv"
d2_path = ".." / DATA_DIR / "dataset2.csv"

# --------------------------------------------------------------------
# Load and prepare data
# --------------------------------------------------------------------
print("Loading data ...")
d1 = pd.read_csv(d1_path)
d2 = pd.read_csv(d2_path)

# Parse time columns
for c in ["start_time","rat_period_start","rat_period_end"]:
    if c in d1.columns:
        d1[c] = pd.to_datetime(d1[c], errors="coerce")
if "time" in d2.columns:
    d2["time"] = pd.to_datetime(d2["time"], errors="coerce")

# Numeric conversion
num_cols1 = ["bat_landing_to_food","seconds_after_rat_arrival",
             "hours_after_sunset","risk","reward"]
for c in num_cols1:
    if c in d1.columns:
        d1[c] = pd.to_numeric(d1[c], errors="coerce")

num_cols2 = ["rat_minutes","rat_arrival_number","bat_landing_number","food_availability"]
for c in num_cols2:
    if c in d2.columns:
        d2[c] = pd.to_numeric(d2[c], errors="coerce")

# Feature: rat present at landing
d1["rat_present"] = np.where(d1["seconds_after_rat_arrival"]>=0,1,0)

# Feature: period_30min for joining datasets
if "start_time" in d1.columns:
    d1["period_30min"] = d1["start_time"].dt.floor("30T")
if "time" in d2.columns:
    d2["period_30min"] = d2["time"].dt.floor("30T")

merged = pd.merge(
    d1,
    d2[["period_30min","rat_arrival_number","rat_minutes",
        "bat_landing_number","food_availability"]],
    on="period_30min", how="left")

# --------------------------------------------------------------------
# Helper: save plot
# --------------------------------------------------------------------
def save_plot(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Saved:", p)

# --------------------------------------------------------------------
# Descriptive summaries
# --------------------------------------------------------------------
summary_stats = d1.describe().T
summary_stats.to_csv(OUT/"summary_stats.csv")

# Seasonal risk proportion
season_risk = (
    d1.groupby("season")["risk"]
    .agg(["count","mean"])
    .rename(columns={"mean":"risk_rate"})
    .reset_index()
)
season_risk.to_csv(OUT/"season_risk.csv", index=False)

# --------------------------------------------------------------------
# 1️⃣ Investigation A revisited – same base analyses
# --------------------------------------------------------------------
fig = plt.figure(figsize=(6,4))
d1["risk"].value_counts().sort_index().plot(kind="bar")
plt.title("Risk-taking behaviour overall")
plt.xlabel("risk (0 = avoidance, 1 = risk-taking)")
plt.ylabel("count")
save_plot(fig,"risk_overall.png")

# t-test / Mann-Whitney for landing time vs rat presence
grp0 = d1[d1["rat_present"]==0]["bat_landing_to_food"].dropna()
grp1 = d1[d1["rat_present"]==1]["bat_landing_to_food"].dropna()
t = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
test_A = {"test":"Mann-Whitney U","U":float(t.statistic),"p":float(t.pvalue)}

# Logistic regression: risk ~ bat_landing_to_food + hours_after_sunset + rat_present
mod_df = d1[["risk","bat_landing_to_food","hours_after_sunset","rat_present"]].dropna()
X = sm.add_constant(mod_df[["bat_landing_to_food","hours_after_sunset","rat_present"]])
y = mod_df["risk"]
logit_A = sm.Logit(y,X).fit(disp=False)
logit_A.save(OUT/"logitA_model.pickle")
open(OUT/"logitA_summary.txt","w").write(logit_A.summary2().as_text())

# --------------------------------------------------------------------
# 2️⃣ Investigation B – seasonal comparisons
# --------------------------------------------------------------------
# Boxplot: bat_landing_to_food by season
fig, ax = plt.subplots(figsize=(7,5))  # explicitly create a figure & axes
d1.boxplot(column="bat_landing_to_food", by="season", ax=ax, grid=False)

ax.set_title("Approach-to-food time by season")
ax.set_ylabel("seconds to approach food")
fig.suptitle("")  # remove default pandas title

save_plot(fig,"landing_to_food_by_season.png")
plt.close(fig)

# Risk rate by season bar
fig = plt.figure(figsize=(6,4))
plt.bar(season_risk["season"], season_risk["risk_rate"])
plt.title("Proportion of risk-taking by season")
plt.xlabel("Season")
plt.ylabel("Mean risk")
save_plot(fig,"risk_by_season.png")

# Kruskal–Wallis (non-parametric ANOVA) for bat_landing_to_food across seasons
groups = [v.dropna().values for k,v in d1.groupby("season")["bat_landing_to_food"]]
kw = stats.kruskal(*groups)
test_B1 = {"test":"Kruskal-Wallis","H":float(kw.statistic),"p":float(kw.pvalue)}

# Logistic regression including season
if "season" in d1.columns:
    df2 = d1[["risk","bat_landing_to_food","hours_after_sunset","rat_present","season"]].dropna()
    df2["season"] = df2["season"].astype("category")
    modelB = logit("risk ~ bat_landing_to_food + hours_after_sunset + rat_present + C(season)", data=df2).fit(disp=False)
    open(OUT/"logitB_summary.txt","w").write(modelB.summary2().as_text())
    modelB.save(OUT/"logitB_model.pickle")
else:
    modelB = None

# --------------------------------------------------------------------
# Season-level correlation between rat and bat activity
# --------------------------------------------------------------------
if "season" in d2.columns:
    corr_by_season = d2.groupby("season")[["rat_arrival_number","bat_landing_number"]].corr().iloc[0::2,-1]
    corr_by_season.to_csv(OUT/"rat_bat_corr_by_season.csv")

# --------------------------------------------------------------------
# Save test results and simple text report
# --------------------------------------------------------------------
results = {
    "Investigation_A": test_A,
    "Investigation_B": {"landing_to_food_KW": test_B1},
}
with open(OUT/"stat_tests.json","w") as f:
    json.dump(results,f,indent=2)

# Short text summary
report = f"""
HIT140 Objective 2 – Investigation A & B Results
-------------------------------------------------
Total bat landings: {len(d1)}
Seasons found: {d1['season'].nunique() if 'season' in d1.columns else 'NA'}

Investigation A:
  Mann-Whitney U test (approach time vs rat presence):
     U = {test_A['U']:.2f}, p = {test_A['p']:.4f}
  Logistic regression summary saved → logitA_summary.txt

Investigation B:
  Kruskal-Wallis test (approach time across seasons):
     H = {test_B1['H']:.2f}, p = {test_B1['p']:.4f}
  Logistic regression with season saved → logitB_summary.txt

Seasonal risk rates:
{season_risk.to_string(index=False)}
"""
open(OUT/"summary_objective3.txt","w", encoding="utf-8").write(report)
print(report)

print("\n✅ All done. Check the 'project_outputs_obj3' folder for:")
print(" - PNG plots")
print(" - CSV summaries")
print(" - TXT / JSON results")
print(" - Logistic regression outputs")

