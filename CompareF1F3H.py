# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:49:07 2026

@author: WangX3
Plot common-cutoff wave statistics for two frames.

Author: WangX3
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

input_file = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
    r"\all_cases_wave_heights_common_fc.csv"
)

frame1_case = "DVN_F1_ADV01"
frame3_case = "DVN_F3_ADV05"

frame1_label = "DVNF1 ADV (20m)"
frame3_label = "DVNF3 ADV (12m)"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    input_file,
    parse_dates=["start_time", "end_time", "mid_time"],
)

df = df.sort_values("mid_time")


# ============================================================
# SPLIT INTO THE TWO CASES
# ============================================================

frame1 = (
    df.loc[df["case_id"] == frame1_case]
    .sort_values("mid_time")
    .copy()
)

frame3 = (
    df.loc[df["case_id"] == frame3_case]
    .sort_values("mid_time")
    .copy()
)


# ============================================================
# CREATE PLOTTING COLUMNS
# ============================================================

for dataset in [frame1, frame3]:
    
    dt = dataset["mid_time"].diff().dt.total_seconds()
    large_gap = dt > 3600

    dataset["ig_ss_ratio"] = (
        dataset["hm0_ig_m"]
        / dataset["hm0_ss_common_fc_m"]
    )

    # dataset.loc[
    #     dataset["hm0_ss_common_fc_m"] <= 0,
    #     "ig_ss_ratio",
    # ] = np.nan
    
    dataset.loc[
        large_gap,
        ["hm0_ig_m", "hm0_ss_common_fc_m", "ig_ss_ratio"],
    ] = np.nan



# ============================================================
# PLOT HIG
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(
    frame1["mid_time"],
    frame1["hm0_ig_m"],
    label=frame1_label,
)

plt.plot(
    frame3["mid_time"],
    frame3["hm0_ig_m"],
    label=frame3_label,
)

plt.xlabel("Time")
plt.ylabel(r"$H_{m0,IG}$ (m)")
plt.title("Infragravity wave height versus time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# ============================================================
# PLOT HSS (COMMON CUTOFF)
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(
    frame1["mid_time"],
    frame1["hm0_ss_common_fc_m"],
    label=frame1_label,
)

plt.plot(
    frame3["mid_time"],
    frame3["hm0_ss_common_fc_m"],
    label=frame3_label,
)

plt.xlabel("Time")
plt.ylabel(r"$H_{m0,SS}$ (m)")
plt.title("Sea-swell wave height (common cutoff) versus time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# ============================================================
# PLOT HIG/HSS
# ============================================================

plt.figure(figsize=(12,5))

plt.plot(
    frame1["mid_time"],
    frame1["ig_ss_ratio"],
    label=frame1_label,
)

plt.plot(
    frame3["mid_time"],
    frame3["ig_ss_ratio"],
    label=frame3_label,
)

plt.xlabel("Time")
plt.ylabel(r"$H_{m0,IG}/H_{m0,SS}$")
plt.title(r"$H_{m0,IG}/H_{m0,SS}$ versus time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ============================================================
# SCATTER: HIG/HSS vs HSS
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    frame1["hm0_ss_common_fc_m"],
    frame1["ig_ss_ratio"],
    s=35,
    alpha=0.7,
    label=frame1_label,
)

plt.scatter(
    frame3["hm0_ss_common_fc_m"],
    frame3["ig_ss_ratio"],
    s=35,
    alpha=0.7,
    label=frame3_label,
)

plt.xlabel(r"$H_{m0,SS,\mathrm{common}}$ (m)")
plt.ylabel(r"$H_{m0,IG}/H_{m0,SS,\mathrm{common}}$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()