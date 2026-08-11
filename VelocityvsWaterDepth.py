# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:00:34 2026

@author: WangX3
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

input_file_vel = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
    r"\all_cases_velocityRMS.csv"
)

input_file_H = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
    r"\all_cases_wave_heights_common_fc.csv"
)

frame1_case = "DVN_F1_ADV01"
frame1_label = "DVN F1 ADV (20 m)"


# ------------------------------------------------------------
# Column names in velocity-RMS CSV
# Change these only if your file uses different names.
# ------------------------------------------------------------

cross_ig_column = "cross_ig_rms"
along_ig_column = "along_ig_rms"

cross_ss_column = "cross_ss_rms"
along_ss_column = "along_ss_rms"

# Water-depth column in wave-height CSV
water_depth_column = "mean_water_depth_m"


# ============================================================
# LOAD DATA
# ============================================================

df_vel = pd.read_csv(
    input_file_vel,
    parse_dates=[
        "start_time",
        "end_time",
        "mid_time",
    ],
)

df_vel = (
    df_vel
    .sort_values("mid_time")
    .reset_index(drop=True)
)

df_H = pd.read_csv(
    input_file_H,
    parse_dates=[
        "start_time",
        "end_time",
        "mid_time",
    ],
)

df_H = (
    df_H
    .sort_values("mid_time")
    .reset_index(drop=True)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_vel_columns = {
    "case_id",
    "mid_time",
    cross_ig_column,
    along_ig_column,
    cross_ss_column,
    along_ss_column,
}

missing_vel = (
    required_vel_columns
    - set(df_vel.columns)
)

if missing_vel:
    raise KeyError(
        "Missing required velocity columns: "
        f"{sorted(missing_vel)}"
    )


required_H_columns = {
    "case_id",
    "mid_time",
    water_depth_column,
}

missing_H = (
    required_H_columns
    - set(df_H.columns)
)

if missing_H:
    raise KeyError(
        "Missing required water-depth columns: "
        f"{sorted(missing_H)}"
    )


# ============================================================
# EXTRACT FRAME 1
# ============================================================

frame1_vel = (
    df_vel.loc[
        df_vel["case_id"] == frame1_case,
        [
            "mid_time",
            cross_ig_column,
            along_ig_column,
            cross_ss_column,
            along_ss_column,
        ],
    ]
    .copy()
    .sort_values("mid_time")
    .reset_index(drop=True)
)

frame1_H = (
    df_H.loc[
        df_H["case_id"] == frame1_case,
        [
            "mid_time",
            water_depth_column,
        ],
    ]
    .copy()
    .sort_values("mid_time")
    .reset_index(drop=True)
)


if frame1_vel.empty:
    raise RuntimeError(
        f"No velocity-RMS data found for {frame1_case}."
    )

if frame1_H.empty:
    raise RuntimeError(
        f"No water-depth data found for {frame1_case}."
    )


# ============================================================
# ALIGN WATER DEPTH WITH VELOCITY RMS
# ============================================================

# Use nearest-time matching because the two products may have
# slightly different burst-center timestamps.
#
# The tolerance can be adjusted if necessary.
merge_tolerance = pd.Timedelta("20min")

frame1 = pd.merge_asof(
    frame1_vel,
    frame1_H,
    on="mid_time",
    direction="nearest",
    tolerance=merge_tolerance,
)

print("\nFrame 1 merged data:")
print(frame1.head())

print(
    "\nVelocity rows:",
    len(frame1_vel),
)

print(
    "Water-depth rows:",
    len(frame1_H),
)

print(
    "Merged rows:",
    len(frame1),
)

print(
    "Rows without matched water depth:",
    frame1[water_depth_column].isna().sum(),
)


# ============================================================
# OPTIONAL: BREAK LINES ACROSS LARGE DATA GAPS
# ============================================================

time_gap_seconds = (
    frame1["mid_time"]
    .diff()
    .dt.total_seconds()
)

large_gap = (
    time_gap_seconds > 3600
)

plot_columns = [
    cross_ig_column,
    along_ig_column,
    cross_ss_column,
    along_ss_column,
    water_depth_column,
]

frame1_plot = frame1.copy()

frame1_plot.loc[
    large_gap,
    plot_columns,
] = np.nan


# ============================================================
# PLOT IG RMS + WATER DEPTH
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
)

# ------------------------------------------------------------
# Cross-shore IG RMS
# ------------------------------------------------------------

ax = axes[0]

line_u = ax.plot(
    frame1_plot["mid_time"],
    frame1_plot[cross_ig_column],
    label="Cross-shore IG RMS",
)

ax.set_ylabel(
    r"$U_{\mathrm{RMS,IG}}$ (m/s)"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    True,
    alpha=0.3,
)

ax_depth = ax.twinx()

line_h = ax_depth.plot(
    frame1_plot["mid_time"],
    frame1_plot[water_depth_column],
    linestyle="--",
    label="Water depth",
)

ax_depth.set_ylabel(
    "Water depth (m)"
)

handles = (
    line_u
    + line_h
)

labels = [
    line.get_label()
    for line in handles
]

ax.legend(
    handles,
    labels,
    loc="upper right",
)

ax.set_title(
    "Cross-shore IG velocity RMS and water depth"
)


# ------------------------------------------------------------
# Alongshore IG RMS
# ------------------------------------------------------------

ax = axes[1]

line_v = ax.plot(
    frame1_plot["mid_time"],
    frame1_plot[along_ig_column],
    label="Alongshore IG RMS",
)

ax.set_ylabel(
    r"$V_{\mathrm{RMS,IG}}$ (m/s)"
)

ax.set_xlabel(
    "Time"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    True,
    alpha=0.3,
)

ax_depth = ax.twinx()

line_h = ax_depth.plot(
    frame1_plot["mid_time"],
    frame1_plot[water_depth_column],
    linestyle="--",
    label="Water depth",
)

ax_depth.set_ylabel(
    "Water depth (m)"
)

handles = (
    line_v
    + line_h
)

labels = [
    line.get_label()
    for line in handles
]

ax.legend(
    handles,
    labels,
    loc="upper right",
)

ax.set_title(
    "Alongshore IG velocity RMS and water depth"
)


fig.suptitle(
    f"Infragravity velocity RMS and water depth ({frame1_label})"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()


# ============================================================
# PLOT SEA-SWELL RMS + WATER DEPTH
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
)

# ------------------------------------------------------------
# Cross-shore SS RMS
# ------------------------------------------------------------

ax = axes[0]

line_u = ax.plot(
    frame1_plot["mid_time"],
    frame1_plot[cross_ss_column],
    label="Cross-shore Sea-Swell RMS",
)

ax.set_ylabel(
    r"$U_{\mathrm{RMS,SS}}$ (m/s)"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    True,
    alpha=0.3,
)

ax_depth = ax.twinx()

line_h = ax_depth.plot(
    frame1_plot["mid_time"],
    frame1_plot[water_depth_column],
    linestyle="--",
    label="Water depth",
)

ax_depth.set_ylabel(
    "Water depth (m)"
)

handles = (
    line_u
    + line_h
)

labels = [
    line.get_label()
    for line in handles
]

ax.legend(
    handles,
    labels,
    loc="upper right",
)

ax.set_title(
    "Cross-shore Sea-Swell velocity RMS and water depth"
)


# ------------------------------------------------------------
# Alongshore SS RMS
# ------------------------------------------------------------

ax = axes[1]

line_v = ax.plot(
    frame1_plot["mid_time"],
    frame1_plot[along_ss_column],
    label="Alongshore Sea-Swell RMS",
)

ax.set_ylabel(
    r"$V_{\mathrm{RMS,SS}}$ (m/s)"
)

ax.set_xlabel(
    "Time"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    True,
    alpha=0.3,
)

ax_depth = ax.twinx()

line_h = ax_depth.plot(
    frame1_plot["mid_time"],
    frame1_plot[water_depth_column],
    linestyle="--",
    label="Water depth",
)

ax_depth.set_ylabel(
    "Water depth (m)"
)

handles = (
    line_v
    + line_h
)

labels = [
    line.get_label()
    for line in handles
]

ax.legend(
    handles,
    labels,
    loc="upper right",
)

ax.set_title(
    "Alongshore Sea-Swell velocity RMS and water depth"
)


fig.suptitle(
    f"Sea-swell velocity RMS and water depth ({frame1_label})"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()

# ============================================================
# SCATTER PLOT RMS - h
# ============================================================
# Convert time to elapsed days for coloring
time_color = (
    frame1["mid_time"]
    - frame1["mid_time"].min()
).dt.total_seconds() / 86400


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 9),
    sharex=True,
)

plot_info = [
    (
        axes[0, 0],
        cross_ig_column,
        r"$U_{\mathrm{RMS,IG}}$ (m/s)",
        "Cross-shore IG",
    ),
    (
        axes[0, 1],
        along_ig_column,
        r"$V_{\mathrm{RMS,IG}}$ (m/s)",
        "Alongshore IG",
    ),
    (
        axes[1, 0],
        cross_ss_column,
        r"$U_{\mathrm{RMS,SS}}$ (m/s)",
        "Cross-shore Sea-Swell",
    ),
    (
        axes[1, 1],
        along_ss_column,
        r"$V_{\mathrm{RMS,SS}}$ (m/s)",
        "Alongshore Sea-Swell",
    ),
]


for ax, column, ylabel, title in plot_info:

    valid = (
        frame1[water_depth_column].notna()
        & frame1[column].notna()
    )

    scatter = ax.scatter(
        frame1.loc[valid, water_depth_column],
        frame1.loc[valid, column],
        c=time_color.loc[valid],
        alpha=0.65,
        s=20,
    )

    ax.set_xlabel("Mean water depth (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


cbar = fig.colorbar(
    scatter,
    ax=axes.ravel().tolist(),
    pad=0.02,
)

cbar.set_label(
    "Days since start of deployment"
)

fig.suptitle(
    f"Velocity RMS versus water depth ({frame1_label})"
)

plt.show()

# ============================================================
# CLASSIFY FLOOD / EBB TIDE
# ============================================================

tidal = frame1.copy()

# ------------------------------------------------------------
# Identify continuous parts of the record
# ------------------------------------------------------------

gap_threshold_seconds = 3600.0

dt_seconds = (
    tidal["mid_time"]
    .diff()
    .dt.total_seconds()
)

new_segment = (
    dt_seconds.isna()
    | (dt_seconds > gap_threshold_seconds)
    | (dt_seconds <= 0)
)

tidal["continuous_segment"] = (
    new_segment.cumsum()
)

for segment_id, segment in tidal.groupby(
    "continuous_segment"
):
    if len(segment) >= 2:
        tidal.loc[
            [segment.index[0], segment.index[-1]],
            "dh_dt_m_per_hr",
        ] = np.nan


# ------------------------------------------------------------
# Smooth h separately inside each continuous segment
# ------------------------------------------------------------

tidal["h_smooth"] = np.nan

for segment_id, segment in tidal.groupby(
    "continuous_segment",
    sort=True,
):

    smoothed = (
        segment[water_depth_column]
        .rolling(
            window=3,
            center=True,
            min_periods=2,
        )
        .mean()
    )

    tidal.loc[
        segment.index,
        "h_smooth",
    ] = smoothed


# ------------------------------------------------------------
# Calculate dh/dt separately inside each continuous segment
# ------------------------------------------------------------

tidal["dh_dt_m_per_hr"] = np.nan

for segment_id, segment in tidal.groupby(
    "continuous_segment",
    sort=True,
):

    valid = (
        segment["mid_time"].notna()
        & segment["h_smooth"].notna()
    )

    sub = segment.loc[valid].copy()

    if len(sub) < 3:
        continue

    time_hours = (
        (
            sub["mid_time"]
            - sub["mid_time"].iloc[0]
        )
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 3600.0
    )

    h = (
        sub["h_smooth"]
        .to_numpy(dtype=float)
    )

    dh_dt = np.gradient(
        h,
        time_hours,
    )

    tidal.loc[
        sub.index,
        "dh_dt_m_per_hr",
    ] = dh_dt


# ------------------------------------------------------------
# Assign tidal stage
# ------------------------------------------------------------

slack_threshold_m_per_hr = 0.01

tidal["tide_stage"] = pd.Series(
    pd.NA,
    index=tidal.index,
    dtype="object",
)

flood_mask = (
    tidal["dh_dt_m_per_hr"]
    > slack_threshold_m_per_hr
)

ebb_mask = (
    tidal["dh_dt_m_per_hr"]
    < -slack_threshold_m_per_hr
)

slack_mask = (
    tidal["dh_dt_m_per_hr"].notna()
    & ~flood_mask
    & ~ebb_mask
)

tidal.loc[
    flood_mask,
    "tide_stage",
] = "flood"

tidal.loc[
    ebb_mask,
    "tide_stage",
] = "ebb"

tidal.loc[
    slack_mask,
    "tide_stage",
] = "slack"


print("\nContinuous segments:")
print(
    tidal.groupby(
        "continuous_segment"
    ).agg(
        start_time=("mid_time", "first"),
        end_time=("mid_time", "last"),
        n=("mid_time", "size"),
    )
)

print("\nTidal-stage counts:")
print(
    tidal["tide_stage"]
    .value_counts(dropna=False)
)

# ============================================================
# VERIFY FLOOD / EBB CLASSIFICATION
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 7),
    sharex=True,
)

# ------------------------------------------------------------
# Water depth
# ------------------------------------------------------------

for segment_id, segment in tidal.groupby(
    "continuous_segment",
    sort=True,
):

    axes[0].plot(
        segment["mid_time"],
        segment[water_depth_column],
        alpha=0.35,
        color="C0",
        label=(
            "Original h"
            if segment_id == tidal["continuous_segment"].min()
            else None
        ),
    )

    axes[0].plot(
        segment["mid_time"],
        segment["h_smooth"],
        color="C1",
        label=(
            "Smoothed h"
            if segment_id == tidal["continuous_segment"].min()
            else None
        ),
    )

axes[0].set_ylabel(
    "Water depth (m)"
)



axes[0].grid(
    True,
    alpha=0.3,
)

axes[0].legend()


# ------------------------------------------------------------
# dh/dt
# ------------------------------------------------------------

for segment_id, segment in tidal.groupby(
    "continuous_segment",
    sort=True,
):

    axes[1].plot(
        segment["mid_time"],
        segment["dh_dt_m_per_hr"],
        color="C0",
    )

axes[1].axhline(
    0,
    linewidth=0.8,
)

axes[1].axhline(
    slack_threshold_m_per_hr,
    linestyle="--",
    label="Slack threshold",
)

axes[1].axhline(
    -slack_threshold_m_per_hr,
    linestyle="--",
)

axes[1].set_xlabel(
    "Time"
)

axes[1].set_ylabel(
    r"$dh/dt$ (m h$^{-1}$)"
)

axes[1].grid(
    True,
    alpha=0.3,
)

axes[1].legend()


fig.suptitle(
    f"Tidal-stage classification ({frame1_label})"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()

# ============================================================
# CORRELATION BY TIDAL STAGE
# ============================================================

variables = {
    "Cross-shore IG":
        cross_ig_column,
    "Alongshore IG":
        along_ig_column,
    "Cross-shore SS":
        cross_ss_column,
    "Alongshore SS":
        along_ss_column,
}

print("\nRMS-depth relationships by tidal stage:")

for label, column in variables.items():

    print("\n" + label)

    for stage in [
        "flood",
        "ebb",
    ]:

        data = tidal.loc[
            tidal["tide_stage"] == stage,
            [
                water_depth_column,
                column,
            ],
        ].dropna()

        if len(data) < 3:
            print(
                stage,
                ": insufficient data",
            )
            continue

        r = data[
            water_depth_column
        ].corr(
            data[column]
        )

        # Simple linear slope:
        # RMS = slope * h + intercept
        slope, intercept = np.polyfit(
            data[water_depth_column],
            data[column],
            deg=1,
        )

        print(
            f"{stage:5s}: "
            f"n={len(data):4d}, "
            f"r={r: .3f}, "
            f"slope={slope: .5f} "
            "(m/s)/m"
        )
        
# ============================================================
# RMS VERSUS DEPTH:
# FLOOD / EBB WITH LINEAR REGRESSION
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 9),
    sharex=True,
)

plot_info = [
    (
        axes[0, 0],
        cross_ig_column,
        r"$U_{\mathrm{RMS,IG}}$ (m/s)",
        "Cross-shore IG",
    ),
    (
        axes[0, 1],
        along_ig_column,
        r"$V_{\mathrm{RMS,IG}}$ (m/s)",
        "Alongshore IG",
    ),
    (
        axes[1, 0],
        cross_ss_column,
        r"$U_{\mathrm{RMS,SS}}$ (m/s)",
        "Cross-shore Sea-Swell",
    ),
    (
        axes[1, 1],
        along_ss_column,
        r"$V_{\mathrm{RMS,SS}}$ (m/s)",
        "Alongshore Sea-Swell",
    ),
]

for ax, column, ylabel, title in plot_info:

    for stage in ["flood", "ebb"]:

        data = tidal.loc[
            tidal["tide_stage"] == stage,
            [
                water_depth_column,
                column,
            ],
        ].dropna()

        # Scatter
        ax.scatter(
            data[water_depth_column],
            data[column],
            alpha=0.25,
            s=15,
            label=stage.capitalize(),
        )

        # Linear regression
        slope, intercept = np.polyfit(
            data[water_depth_column],
            data[column],
            deg=1,
        )

        r = data[
            water_depth_column
        ].corr(
            data[column]
        )

        h_fit = np.linspace(
            data[water_depth_column].min(),
            data[water_depth_column].max(),
            100,
        )

        rms_fit = (
            slope * h_fit
            + intercept
        )

        ax.plot(
            h_fit,
            rms_fit,
            linewidth=2,
            label=(
                f"{stage.capitalize()} fit "
                f"(r={r:.2f})"
            ),
        )

    ax.set_xlabel(
        "Mean water depth (m)"
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()


fig.suptitle(
    f"Flood–ebb dependence of velocity RMS "
    f"({frame1_label})"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()