# -*- coding: utf-8 -*-
"""
Compare ADCP velocity RMS, tidal current, water depth,
and pressure-derived IG/SS wave energy for F1 and F3.

Velocity/current input
----------------------
all_cases_ADCP_current_wave_axis.csv

Wave-height/water-depth input
-----------------------------
all_cases_wave_heights_common_fc.csv

For each frame:
1. Select the highest processed ADCP cell.
2. Match ADCP burst statistics with pressure/wave statistics by time.
3. Plot IG velocity RMS vs water depth.
4. Plot SS velocity RMS vs water depth.
5. Plot RH = m0_IG/m0_SS vs water depth.
6. Plot:
       Ru = |U_RMS,IG|^2 / |U_RMS,SS|^2 vs Ucurrent
       RH = m0_IG/m0_SS vs Ucurrent
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
    r"\all_cases_ADCP_current_wave_axis.csv"
)

input_file_H = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
    r"\all_cases_wave_heights_common_fc.csv"
)

frames_to_plot = ["F1", "F3"]

frame_labels = {
    "F1": "Frame 1",
    "F3": "Frame 3",
}

merge_tolerance = pd.Timedelta("20min")
plot_gap_seconds = 3600.0

# PLOT TIME WINDOW
plot_start = pd.Timestamp("2018-04-05 00:00:00", tz="UTC")
plot_end = pd.Timestamp("2018-05-15 00:00:00", tz="UTC")  


# ============================================================
# COLUMN NAMES
# ============================================================

cross_ig_column = "cross_ig_rms_m_s"
along_ig_column = "along_ig_rms_m_s"
horizontal_ig_column = "horizontal_ig_rms_m_s"

cross_ss_column = "cross_ss_rms_m_s"
along_ss_column = "along_ss_rms_m_s"
horizontal_ss_column = "horizontal_ss_rms_m_s"

current_speed_column = "depth_avg_current_speed_m_s"

cell_column = "cell_number"
height_column = "height_relative_to_frame_bottom_m"

water_depth_column = "mean_water_depth_m"
hm0_ig_column = "hm0_ig_m"
hm0_ss_column = "hm0_ss_m"


# ============================================================
# FUNCTIONS
# ============================================================

def extract_frame_id(series):
    return (
        series.astype(str)
        .str.extract(r"(F\d+)", expand=False)
        .str.upper()
    )


def normalize_time(series):
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")


def break_plot_gaps(data, columns):
    output = data.copy()

    large_gap = (
        output["mid_time"]
        .diff()
        .dt.total_seconds()
        > plot_gap_seconds
    )

    output.loc[
        large_gap,
        columns,
    ] = np.nan

    return output


# ============================================================
# LOAD DATA
# ============================================================

print("Reading ADCP velocity/current statistics:")
print(input_file_vel)

df_vel = pd.read_csv(
    input_file_vel
)

print("\nReading wave-height/water-depth statistics:")
print(input_file_H)

df_H = pd.read_csv(
    input_file_H
)


# ============================================================
# NORMALIZE TIME
# ============================================================

for df in [df_vel, df_H]:
    for column in [
        "start_time",
        "end_time",
        "mid_time",
    ]:
        if column in df.columns:
            df[column] = normalize_time(
                df[column]
            )


# ============================================================
# FRAME IDs
# ============================================================

df_vel["frame_id"] = extract_frame_id(
    df_vel["case_id"]
)

df_H["frame_id"] = extract_frame_id(
    df_H["case_id"]
)

print("\nADCP cases:")
print(
    df_vel[
        ["case_id", "frame_id"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nWave-height cases:")
print(
    df_H[
        ["case_id", "frame_id"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_vel_columns = {
    "case_id",
    "frame_id",
    "mid_time",
    cell_column,
    height_column,
    cross_ig_column,
    along_ig_column,
    horizontal_ig_column,
    cross_ss_column,
    along_ss_column,
    horizontal_ss_column,
    current_speed_column,
}

required_H_columns = {
    "case_id",
    "frame_id",
    "mid_time",
    water_depth_column,
    hm0_ig_column,
    hm0_ss_column,
}

missing_vel = (
    required_vel_columns
    - set(df_vel.columns)
)

missing_H = (
    required_H_columns
    - set(df_H.columns)
)

if missing_vel:
    raise KeyError(
        "Missing ADCP velocity columns: "
        f"{sorted(missing_vel)}"
    )

if missing_H:
    raise KeyError(
        "Missing wave-height columns: "
        f"{sorted(missing_H)}"
    )


# ============================================================
# BUILD MATCHED DATA FOR F1 AND F3
# ============================================================

matched = {}

for frame_id in frames_to_plot:

    # --------------------------------------------------------
    # ADCP data for this frame
    # --------------------------------------------------------

    vel = (
        df_vel.loc[
            df_vel["frame_id"] == frame_id
        ]
        .copy()
    )

    if vel.empty:
        print(
            f"\nNo ADCP velocity data for {frame_id}."
        )
        continue


    # --------------------------------------------------------
    # Select highest processed ADCP cell
    # --------------------------------------------------------

    cell_heights = (
        vel[
            [
                cell_column,
                height_column,
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    highest = cell_heights.loc[
        cell_heights[
            height_column
        ].idxmax()
    ]

    selected_cell = int(
        highest[
            cell_column
        ]
    )

    selected_height = float(
        highest[
            height_column
        ]
    )

    print(
        f"\n{frame_id}: using ADCP cell "
        f"{selected_cell}, "
        f"z={selected_height:.3f} m"
    )

    vel = (
        vel.loc[
            vel[
                cell_column
            ]
            == selected_cell,
            [
                "mid_time",
                cross_ig_column,
                along_ig_column,
                horizontal_ig_column,
                cross_ss_column,
                along_ss_column,
                horizontal_ss_column,
                current_speed_column,
            ],
        ]
        .dropna(
            subset=["mid_time"]
        )
        .sort_values("mid_time")
        .drop_duplicates(
            subset="mid_time"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Pressure/wave data for same frame
    # --------------------------------------------------------

    wave = (
        df_H.loc[
            df_H["frame_id"] == frame_id,
            [
                "mid_time",
                water_depth_column,
                hm0_ig_column,
                hm0_ss_column,
            ],
        ]
        .dropna(
            subset=["mid_time"]
        )
        .sort_values("mid_time")
        .drop_duplicates(
            subset="mid_time"
        )
        .reset_index(drop=True)
    )

    if wave.empty:
        print(
            f"No wave/water-depth data for {frame_id}."
        )
        continue


    # --------------------------------------------------------
    # Match nearest burst times
    # --------------------------------------------------------

    data = pd.merge_asof(
        vel,
        wave,
        on="mid_time",
        direction="nearest",
        tolerance=merge_tolerance,
    )


    # ========================================================
    # RATIOS
    # ========================================================

    # Velocity variance ratio
    #
    # Ru = |U_RMS,IG|^2 / |U_RMS,SS|^2

    data["Ru"] = (
        data[
            horizontal_ig_column
        ] ** 2
        / data[
            horizontal_ss_column
        ] ** 2
    )


    # Surface-elevation variance ratio
    #
    # Hm0 = 4 sqrt(m0)
    #
    # therefore:
    #
    # m0_IG / m0_SS
    # = Hm0_IG^2 / Hm0_SS^2

    data["RH"] = (
        data[
            hm0_ig_column
        ] ** 2
        / data[
            hm0_ss_column
        ] ** 2
    )


    # Remove infinities caused by zero denominator
    data[
        ["Ru", "RH"]
    ] = (
        data[
            ["Ru", "RH"]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


    matched[
        frame_id
    ] = data


    print(
        f"{frame_id}: "
        f"{len(vel)} velocity bursts, "
        f"{len(wave)} wave bursts, "
        f"{data[water_depth_column].notna().sum()} matched"
    )


# ============================================================
# DERIVED VARIABLES
# ============================================================

for frame_id, d in matched.items():
    d["Ru"] = d[horizontal_ig_column]**2 / d[horizontal_ss_column]**2
    d["m0_IG"] = d[hm0_ig_column]**2 / 16 *10000
    d["m0_SS"] = d[hm0_ss_column]**2 / 16 *10000
    d["RH"] = d["m0_IG"] / d["m0_SS"]
    d[["Ru", "RH"]] = d[["Ru", "RH"]].replace([np.inf, -np.inf], np.nan)

# ============================================================
# FIGURES 1-2
# |URMS,IG|, |URMS,SS|, Ru, Ucurrent, h VS TIME
# ============================================================

for frame_id in ["F1", "F3"]:
    if frame_id not in matched:
        continue

    d = matched[frame_id].loc[
    (matched[frame_id]["mid_time"] >= plot_start) &
    (matched[frame_id]["mid_time"] < plot_end)
].sort_values("mid_time").copy()
    cols = [
        horizontal_ig_column,
        horizontal_ss_column,
        "Ru",
        current_speed_column,
        water_depth_column,
    ]
    d = break_plot_gaps(d, cols)

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    variables = [
        (horizontal_ig_column, r"$|\mathbf{U}_{\mathrm{RMS,IG}}|$ (m/s)"),
        (horizontal_ss_column, r"$|\mathbf{U}_{\mathrm{RMS,SS}}|$ (m/s)"),
        ("Ru", r"$R_u=|\mathbf{U}_{\mathrm{RMS,IG}}|^2/|\mathbf{U}_{\mathrm{RMS,SS}}|^2$"),
        (current_speed_column, r"$U_{\mathrm{current}}$ (m/s)"),
        (water_depth_column, r"$h$ (m)"),
    ]

    for ax, (column, ylabel) in zip(axes, variables):
        ax.plot(d["mid_time"], d[column])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[:4]:
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Velocity IG/SS modulation ({frame_labels[frame_id]})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ============================================================
# FIGURES 3-4
# m0IG, m0SS, RH, Ucurrent, h VS TIME
# ============================================================

for frame_id in ["F1", "F3"]:
    if frame_id not in matched:
        continue

    d = matched[frame_id].loc[
    (matched[frame_id]["mid_time"] >= plot_start) &
    (matched[frame_id]["mid_time"] < plot_end)
].sort_values("mid_time").copy()
    cols = [
        "m0_IG",
        "m0_SS",
        "RH",
        current_speed_column,
        water_depth_column,
    ]
    d = break_plot_gaps(d, cols)

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    variables = [
        ("m0_IG", r"$m_{0,\mathrm{IG}}$ (cm$^2$)"),
        ("m0_SS", r"$m_{0,\mathrm{SS}}$ (cm$^2$)"),
        ("RH", r"$R_H=m_{0,\mathrm{IG}}/m_{0,\mathrm{SS}}$"),
        (current_speed_column, r"$U_{\mathrm{current}}$ (m/s)"),
        (water_depth_column, r"$h$ (m)"),
    ]

    for i, (ax, (column, ylabel)) in enumerate(zip(axes, variables)):
        ax.plot(d["mid_time"], d[column])
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

        # Log-y scale for m0 panels only
        if i < 2:
            ax.set_yscale("log")
        elif i < 4:
            ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time")

    fig.suptitle(
        f"Surface-wave IG/SS modulation ({frame_labels[frame_id]})"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
# ============================================================
# FIGURE 5
# Ru-Ucurrent AND RH-Ucurrent, F1/F3
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

for j, frame_id in enumerate(["F1", "F3"]):
    if frame_id not in matched:
        continue

    d = matched[frame_id]

    x = d[[current_speed_column, "Ru"]].dropna()
    axes[0, j].scatter(
        x[current_speed_column], x["Ru"],
        s=18, alpha=0.5
    )
    axes[0, j].set_ylabel(r"$R_u$")
    axes[0, j].set_title(frame_labels[frame_id])
    axes[0, j].set_ylim(bottom=0)
    axes[0, j].grid(True, alpha=0.3)

    x = d[[current_speed_column, "RH"]].dropna()
    axes[1, j].scatter(
        x[current_speed_column], x["RH"],
        s=18, alpha=0.5
    )
    axes[1, j].set_xlabel(r"$U_{\mathrm{current}}$ (m/s)")
    axes[1, j].set_ylabel(r"$R_H$")
    axes[1, j].set_ylim(bottom=0)
    axes[1, j].grid(True, alpha=0.3)

fig.suptitle(
    "IG/SS ratios versus depth-averaged current speed\n"
    f"{plot_start} to {plot_end}"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()