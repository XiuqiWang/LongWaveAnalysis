# -*- coding: utf-8 -*-
"""
Compare ADCP velocity RMS, depth-averaged tidal current,
ADCP water depth, and pressure-derived IG/SS surface-wave energy
for F1 and F3.

Data sources
------------
1. all_cases_ADCP_current_wave_axis.csv
       -> burst-level ADCP velocity RMS
          cross_ig_rms_m_s
          along_ig_rms_m_s
          horizontal_ig_rms_m_s
          cross_ss_rms_m_s
          along_ss_rms_m_s
          horizontal_ss_rms_m_s

2. all_cases_wave_heights_common_fc.csv
       -> pressure-derived wave statistics
          hm0_ig_m
          hm0_ss_m

3. DVN_F1/F3_ADCP_depth_averaged_current.csv
       -> 10-min processed depth-averaged ADCP current
          depth_avg_cross_shore_m_s
          depth_avg_alongshore_m_s
          depth_avg_current_speed_m_s
          water_depth_m

Workflow
--------
For each frame:
1. Select the highest processed high-resolution ADCP cell.
2. Match velocity RMS to pressure-derived wave statistics.
3. Match the new depth-averaged current/water-depth time series.
4. Calculate:
       Ru = U_RMS,IG^2 / U_RMS,SS^2
       RH = m0_IG / m0_SS
5. Plot:
       Figure 1/2:
           U_RMS,IG
           U_RMS,SS
           Ru
           depth-averaged current speed
           ADCP water depth

       Figure 3/4:
           m0_IG
           m0_SS
           RH
           depth-averaged current speed
           ADCP water depth

       Figure 5:
           Ru versus depth-averaged current speed
           RH versus depth-averaged current speed
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch, csd


# ============================================================
# USER SETTINGS
# ============================================================

spectra_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)

processed_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)


# ------------------------------------------------------------
# Velocity RMS from high-resolution ADCP analysis
# ------------------------------------------------------------

input_file_vel = (
    spectra_folder
    / "all_cases_ADCP_current_wave_axis.csv"
)


# ------------------------------------------------------------
# Pressure-derived IG/SS wave statistics
#
# Water depth from this file is NOT used anymore.
# ------------------------------------------------------------

input_file_wave = (
    spectra_folder
    / "all_cases_wave_heights_common_fc.csv"
)


# ------------------------------------------------------------
# NEW depth-averaged ADCP current + ADCP water depth
# ------------------------------------------------------------

depth_averaged_files = {
    "F1": (
        processed_folder
        / "DVN_F1_ADCP_upward_depth_averaged_current.csv"
    ),

    "F3": (
        processed_folder
        / "DVN_F3_ADCP_upward_depth_averaged_current.csv"
    ),
}


frames_to_plot = [
    "F1",
    "F3",
]

frame_labels = {
    "F1": "Frame 1",
    "F3": "Frame 3",
}


# ------------------------------------------------------------
# Time matching
# ------------------------------------------------------------

# High-resolution ADCP bursts and pressure-wave bursts
wave_merge_tolerance = pd.Timedelta(
    "20min"
)

# New current data have nominal 10-min resolution.
# A 10-min tolerance should normally be sufficient.
current_merge_tolerance = pd.Timedelta(
    "10min"
)


# ------------------------------------------------------------
# Plot-gap threshold
# ------------------------------------------------------------

plot_gap_seconds = 3600.0


# ------------------------------------------------------------
# Plot time window
# ------------------------------------------------------------

plot_start = pd.Timestamp(
    "2018-04-05 00:00:00",
    tz="UTC",
)

plot_end = pd.Timestamp(
    "2018-05-15 00:00:00",
    tz="UTC",
)


# ============================================================
# COLUMN NAMES
# ============================================================

# ------------------------------------------------------------
# High-resolution ADCP velocity RMS
# ------------------------------------------------------------

cross_ig_column = (
    "cross_ig_rms_m_s"
)

along_ig_column = (
    "along_ig_rms_m_s"
)

horizontal_ig_column = (
    "horizontal_ig_rms_m_s"
)

cross_ss_column = (
    "cross_ss_rms_m_s"
)

along_ss_column = (
    "along_ss_rms_m_s"
)

horizontal_ss_column = (
    "horizontal_ss_rms_m_s"
)

cell_column = (
    "cell_number"
)

height_column = (
    "height_relative_to_frame_bottom_m"
)


# ------------------------------------------------------------
# Pressure-derived wave statistics
# ------------------------------------------------------------

hm0_ig_column = (
    "hm0_ig_m"
)

hm0_ss_column = (
    "hm0_ss_m"
)


# ------------------------------------------------------------
# NEW depth-averaged ADCP current
# ------------------------------------------------------------

current_cross_column = (
    "depth_avg_cross_shore_m_s"
)

current_along_column = (
    "depth_avg_alongshore_m_s"
)

current_speed_column = (
    "depth_avg_current_speed_m_s"
)

# Water depth now comes from the new ADCP current file
water_depth_column = (
    "water_depth_m"
)


# ============================================================
# FUNCTIONS
# ============================================================

def extract_frame_id(series):
    """Extract F1, F3, etc. from case_id."""

    return (
        series
        .astype(str)
        .str.extract(
            r"(F\d+)",
            expand=False,
        )
        .str.upper()
    )


def normalize_time(series):
    """Force timestamp to datetime64[ns, UTC]."""

    return (
        pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )
        .astype(
            "datetime64[ns, UTC]"
        )
    )


def break_plot_gaps(
    data,
    columns,
):
    """
    Insert NaNs after temporal gaps so matplotlib
    does not connect separate measurement periods.
    """

    output = (
        data.copy()
    )

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
# READ HIGH-RESOLUTION ADCP VELOCITY RMS
# ============================================================

print(
    "Reading ADCP velocity RMS statistics:"
)

print(
    input_file_vel
)

df_vel = pd.read_csv(
    input_file_vel
)

df_vel["mid_time"] = (
    normalize_time(
        df_vel["mid_time"]
    )
)

if "case_id" not in df_vel.columns:
    raise KeyError(
        "Velocity file does not contain 'case_id'."
    )

df_vel["frame_id"] = (
    extract_frame_id(
        df_vel["case_id"]
    )
)


# ============================================================
# READ PRESSURE-DERIVED WAVE STATISTICS
# ============================================================

print(
    "\nReading pressure-derived wave statistics:"
)

print(
    input_file_wave
)

df_wave = pd.read_csv(
    input_file_wave
)

df_wave["mid_time"] = (
    normalize_time(
        df_wave["mid_time"]
    )
)

if "case_id" not in df_wave.columns:
    raise KeyError(
        "Wave statistics file does not contain 'case_id'."
    )

df_wave["frame_id"] = (
    extract_frame_id(
        df_wave["case_id"]
    )
)


# ============================================================
# CHECK REQUIRED HIGH-RESOLUTION VELOCITY COLUMNS
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
}

missing_vel = (
    required_vel_columns
    - set(df_vel.columns)
)

if missing_vel:

    raise KeyError(
        "Missing ADCP velocity columns: "
        f"{sorted(missing_vel)}"
    )


# ============================================================
# CHECK REQUIRED PRESSURE/WAVE COLUMNS
# ============================================================

required_wave_columns = {
    "case_id",
    "frame_id",
    "mid_time",
    hm0_ig_column,
    hm0_ss_column,
}

missing_wave = (
    required_wave_columns
    - set(df_wave.columns)
)

if missing_wave:

    raise KeyError(
        "Missing pressure-wave columns: "
        f"{sorted(missing_wave)}"
    )


# ============================================================
# READ NEW DEPTH-AVERAGED CURRENT FILES
# ============================================================

depth_averaged = {}

for frame_id in frames_to_plot:

    current_file = (
        depth_averaged_files[
            frame_id
        ]
    )

    print(
        f"\nReading {frame_id} "
        "depth-averaged current:"
    )

    print(
        current_file
    )

    if not current_file.exists():

        print(
            f"WARNING: file does not exist for "
            f"{frame_id}: {current_file}"
        )

        continue

    current = pd.read_csv(
        current_file
    )

    # New file uses "time", not "mid_time"
    if "time" not in current.columns:

        raise KeyError(
            f"{current_file.name} "
            "does not contain 'time'."
        )

    current["time"] = (
        normalize_time(
            current["time"]
        )
    )

    required_current_columns = {
        "time",
        current_cross_column,
        current_along_column,
        current_speed_column,
        water_depth_column,
    }

    missing_current = (
        required_current_columns
        - set(current.columns)
    )

    if missing_current:

        raise KeyError(
            f"Missing columns in "
            f"{current_file.name}: "
            f"{sorted(missing_current)}"
        )

    current = (
        current[
            [
                "time",
                current_cross_column,
                current_along_column,
                current_speed_column,
                water_depth_column,
            ]
        ]
        .dropna(
            subset=["time"]
        )
        .sort_values(
            "time"
        )
        .drop_duplicates(
            subset=["time"]
        )
        .reset_index(
            drop=True
        )
    )

    # Rename for merge_asof
    current = current.rename(
        columns={
            "time":
                "current_time"
        }
    )

    depth_averaged[
        frame_id
    ] = current

    print(
        f"{frame_id}: "
        f"{len(current)} current records"
    )

    print(
        "Start:",
        current[
            "current_time"
        ].min(),
    )

    print(
        "End:",
        current[
            "current_time"
        ].max(),
    )


# ============================================================
# BUILD MATCHED DATA
# ============================================================

matched = {}


for frame_id in frames_to_plot:

    if (
        frame_id
        not in depth_averaged
    ):

        print(
            f"\nSkipping {frame_id}: "
            "no depth-averaged current file."
        )

        continue


    # ========================================================
    # HIGH-RESOLUTION ADCP VELOCITY RMS
    # ========================================================

    vel = (
        df_vel.loc[
            df_vel[
                "frame_id"
            ]
            == frame_id
        ]
        .copy()
    )

    if vel.empty:

        print(
            f"No ADCP velocity RMS data "
            f"for {frame_id}."
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

    highest = (
        cell_heights.loc[
            cell_heights[
                height_column
            ]
            .idxmax()
        ]
    )

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
        f"\n{frame_id}: using HR ADCP cell "
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
            ],
        ]
        .dropna(
            subset=[
                "mid_time"
            ]
        )
        .sort_values(
            "mid_time"
        )
        .drop_duplicates(
            subset=[
                "mid_time"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # PRESSURE-DERIVED IG/SS WAVE STATISTICS
    # ========================================================

    wave = (
        df_wave.loc[
            df_wave[
                "frame_id"
            ]
            == frame_id,
            [
                "mid_time",
                hm0_ig_column,
                hm0_ss_column,
            ],
        ]
        .dropna(
            subset=[
                "mid_time"
            ]
        )
        .sort_values(
            "mid_time"
        )
        .drop_duplicates(
            subset=[
                "mid_time"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if wave.empty:

        print(
            f"No pressure/wave data "
            f"for {frame_id}."
        )

        continue


    # ========================================================
    # MATCH VELOCITY RMS TO PRESSURE WAVE STATISTICS
    # ========================================================

    data = pd.merge_asof(
        vel,
        wave,
        on="mid_time",
        direction="nearest",
        tolerance=
            wave_merge_tolerance,
    )


    # ========================================================
    # MATCH NEW DEPTH-AVERAGED CURRENT AND WATER DEPTH
    # ========================================================

    current = (
        depth_averaged[
            frame_id
        ]
        .copy()
        .sort_values(
            "current_time"
        )
    )

    data = (
        data
        .sort_values(
            "mid_time"
        )
    )

    data = pd.merge_asof(
        data,
        current,
        left_on="mid_time",
        right_on="current_time",
        direction="nearest",
        tolerance=
            current_merge_tolerance,
    )


    # ========================================================
    # RATIOS
    # ========================================================

    # Velocity variance ratio:
    #
    # Ru = U_RMS,IG^2 / U_RMS,SS^2

    data["Ru"] = (
        data[
            horizontal_ig_column
        ] ** 2
        /
        data[
            horizontal_ss_column
        ] ** 2
    )


    # Surface-elevation variances:
    #
    # Hm0 = 4 sqrt(m0)
    #
    # m0 = Hm0^2 / 16
    #
    # Convert m^2 -> cm^2:
    # multiply by 10000.

    data["m0_IG"] = (
        data[
            hm0_ig_column
        ] ** 2
        / 16.0
        * 10000.0
    )

    data["m0_SS"] = (
        data[
            hm0_ss_column
        ] ** 2
        / 16.0
        * 10000.0
    )


    data["RH"] = (
        data["m0_IG"]
        /
        data["m0_SS"]
    )


    data[
        [
            "Ru",
            "RH",
        ]
    ] = (
        data[
            [
                "Ru",
                "RH",
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


    matched[
        frame_id
    ] = data


    print(
        f"{frame_id}: "
        f"{len(vel)} HR velocity bursts, "
        f"{len(wave)} pressure-wave bursts"
    )

    print(
        "Matched pressure-wave records:",
        data[
            hm0_ig_column
        ]
        .notna()
        .sum(),
    )

    print(
        "Matched depth-averaged current records:",
        data[
            current_speed_column
        ]
        .notna()
        .sum(),
    )

    print(
        "Matched ADCP water-depth records:",
        data[
            water_depth_column
        ]
        .notna()
        .sum(),
    )


# ============================================================
# FIGURES 1-2
#
# U_RMS,IG
# U_RMS,SS
# Ru
# DEPTH-AVERAGED CURRENT SPEED
# ADCP WATER DEPTH
# ============================================================

for frame_id in [
    "F1",
    "F3",
]:

    if frame_id not in matched:
        continue


    d = (
        matched[
            frame_id
        ]
        .loc[
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                >= plot_start
            )
            &
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                < plot_end
            )
        ]
        .sort_values(
            "mid_time"
        )
        .copy()
    )


    columns_to_break = [
        horizontal_ig_column,
        horizontal_ss_column,
        "Ru",
        current_speed_column,
        water_depth_column,
    ]

    d = break_plot_gaps(
        d,
        columns_to_break,
    )


    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            13,
            11,
        ),
        sharex=True,
    )


    variables = [
        (
            horizontal_ig_column,
            r"$U_{h,\mathrm{RMS,IG}}$ (m/s)",
        ),

        (
            horizontal_ss_column,
            r"$U_{h,\mathrm{RMS,SS}}$ (m/s)",
        ),

        (
            "Ru",
            r"$R_U="
            r"U_{h,\mathrm{RMS,IG}}^2/"
            r"U_{h,\mathrm{RMS,SS}}^2$",
        ),

        (
            current_speed_column,
            r"$|\mathbf{U}_{DA}|$ (m/s)",
        ),

        (
            water_depth_column,
            r"$h_{\mathrm{ADCP}}$ (m)",
        ),
    ]


    for (
        ax,
        (
            column,
            ylabel,
        ),
    ) in zip(
        axes,
        variables,
    ):

        ax.plot(
            d["mid_time"],
            d[column],
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            True,
            alpha=0.3,
        )


    # Nonnegative quantities
    for ax in axes[:4]:

        ax.set_ylim(
            bottom=0
        )


    axes[-1].set_xlabel(
        "Time"
    )


    fig.suptitle(
        "Velocity IG/SS modulation and "
        "depth-averaged tidal current\n"
        f"{frame_labels[frame_id]}, "
        f"{plot_start:%Y-%m-%d} to "
        f"{plot_end:%Y-%m-%d}"
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    plt.show()


# ============================================================
# FIGURES 3-4
#
# m0IG
# m0SS
# RH
# DEPTH-AVERAGED CURRENT SPEED
# ADCP WATER DEPTH
# ============================================================

for frame_id in [
    "F1",
    "F3",
]:

    if frame_id not in matched:
        continue


    d = (
        matched[
            frame_id
        ]
        .loc[
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                >= plot_start
            )
            &
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                < plot_end
            )
        ]
        .sort_values(
            "mid_time"
        )
        .copy()
    )


    columns_to_break = [
        "m0_IG",
        "m0_SS",
        "RH",
        current_speed_column,
        water_depth_column,
    ]


    d = break_plot_gaps(
        d,
        columns_to_break,
    )


    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            13,
            11,
        ),
        sharex=True,
    )


    variables = [
        (
            "m0_IG",
            r"$m_{0,\mathrm{IG}}$ "
            r"(cm$^2$)",
        ),

        (
            "m0_SS",
            r"$m_{0,\mathrm{SS}}$ "
            r"(cm$^2$)",
        ),

        (
            "RH",
            r"$R_H="
            r"m_{0,\mathrm{IG}}/"
            r"m_{0,\mathrm{SS}}$",
        ),

        (
            current_speed_column,
            r"$|\mathbf{U}_{DA}|$ (m/s)",
        ),

        (
            water_depth_column,
            r"$h_{\mathrm{ADCP}}$ (m)",
        ),
    ]


    for i, (
        ax,
        (
            column,
            ylabel,
        ),
    ) in enumerate(
        zip(
            axes,
            variables,
        )
    ):

        ax.plot(
            d["mid_time"],
            d[column],
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )


        # m0 panels
        if i < 2:

            ax.set_yscale(
                "log"
            )


        # RH and current speed
        elif i < 4:

            ax.set_ylim(
                bottom=0
            )


    axes[-1].set_xlabel(
        "Time"
    )


    fig.suptitle(
        "Surface-wave IG/SS modulation and "
        "depth-averaged tidal current\n"
        f"{frame_labels[frame_id]}, "
        f"{plot_start:%Y-%m-%d} to "
        f"{plot_end:%Y-%m-%d}"
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    plt.show()


# ============================================================
# FIGURE 5
#
# Ru VS DEPTH-AVERAGED CURRENT SPEED
# RH VS DEPTH-AVERAGED CURRENT SPEED
# F1 / F3
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        10,
        8,
    ),
    sharex="col",
)


for j, frame_id in enumerate(
    [
        "F1",
        "F3",
    ]
):

    if frame_id not in matched:
        continue


    # Use exactly the same requested time window
    # as the time-series figures.

    d = (
        matched[
            frame_id
        ]
        .loc[
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                >= plot_start
            )
            &
            (
                matched[
                    frame_id
                ][
                    "mid_time"
                ]
                < plot_end
            )
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Ru versus current speed
    # --------------------------------------------------------

    x = (
        d[
            [
                current_speed_column,
                "Ru",
            ]
        ]
        .dropna()
    )


    axes[
        0,
        j,
    ].scatter(
        x[
            current_speed_column
        ],
        x[
            "Ru"
        ],
        s=18,
        alpha=0.5,
    )


    axes[
        0,
        j,
    ].set_ylabel(
        r"$R_U$"
    )

    axes[
        0,
        j,
    ].set_title(
        frame_labels[
            frame_id
        ]
    )

    axes[
        0,
        j,
    ].set_ylim(
        bottom=0
    )

    axes[
        0,
        j,
    ].grid(
        True,
        alpha=0.3,
    )


    # --------------------------------------------------------
    # RH versus current speed
    # --------------------------------------------------------

    x = (
        d[
            [
                current_speed_column,
                "RH",
            ]
        ]
        .dropna()
    )


    axes[
        1,
        j,
    ].scatter(
        x[
            current_speed_column
        ],
        x[
            "RH"
        ],
        s=18,
        alpha=0.5,
    )


    axes[
        1,
        j,
    ].set_xlabel(
        r"Depth-averaged current speed "
        r"$|\mathbf{U}_{DA}|$ (m/s)"
    )

    axes[
        1,
        j,
    ].set_ylabel(
        r"$R_H$"
    )

    axes[
        1,
        j,
    ].set_ylim(
        bottom=0
    )

    axes[
        1,
        j,
    ].grid(
        True,
        alpha=0.3,
    )


fig.suptitle(
    "IG/SS ratios versus depth-averaged current speed\n"
    f"{plot_start:%Y-%m-%d} to "
    f"{plot_end:%Y-%m-%d}"
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95,
    ]
)

plt.show()

# ============================================================
# TIDAL-TIMESCALE COHERENCE + PHASE
# ============================================================

grid_minutes = 10
max_interp_gap = 6          # 1 hour
welch_days = 5
overlap_fraction = 0.5

plot_fmin_cpd = 0.5
plot_fmax_cpd = 4.0

# O1_cpd = 0.9295
# K1_cpd = 1.0027
M2_cpd = 1.9323
S2_cpd = 2.0000

response_vars = {
    horizontal_ig_column: r"$U_{h,\mathrm{RMS,IG}}$",
    horizontal_ss_column: r"$U_{h,\mathrm{RMS,SS}}$",
    "Ru": r"$R_U$",
    "m0_IG": r"$m_{0,\mathrm{IG}}$",
    "m0_SS": r"$m_{0,\mathrm{SS}}$",
    "RH": r"$R_H$",
}

reference_vars = {
    current_along_column: r"$U_{\mathrm{DA,along}}$",
    current_speed_column: r"$|U_{\mathrm{DA}}|$",
    water_depth_column: r"$h$",
}


def interp_short_gaps(s, max_gap):
    """Interpolate only complete internal gaps <= max_gap samples."""
    s = pd.Series(s, dtype=float).copy()
    miss = s.isna()

    if not miss.any():
        return s

    candidate = s.interpolate(
        method="linear",
        limit_area="inside",
    )

    groups = miss.ne(miss.shift()).cumsum()

    for g in groups[miss].unique():
        idx = groups.index[(groups == g) & miss]

        if len(idx) > max_gap:
            continue

        if idx[0] == s.index[0] or idx[-1] == s.index[-1]:
            continue

        if (
            np.isfinite(s.loc[idx[0] - 1])
            and np.isfinite(s.loc[idx[-1] + 1])
        ):
            s.loc[idx] = candidate.loc[idx]

    return s


def make_10min_grid(d):
    """Put all variables onto one regular 10-min grid."""
    cols = list(response_vars) + list(reference_vars)

    x = (
        d.loc[
            (d["mid_time"] >= plot_start)
            & (d["mid_time"] < plot_end),
            ["mid_time"] + cols,
        ]
        .dropna(subset=["mid_time"])
        .sort_values("mid_time")
        .set_index("mid_time")
        .resample(f"{grid_minutes}min")
        .mean()
    )

    for col in cols:
        x[col] = interp_short_gaps(
            x[col].reset_index(drop=True),
            max_interp_gap,
        ).to_numpy()

    return x.reset_index()


def spectral_pair(reference, response):
    """
    Cross-spectrum with:
        phase = response phase - reference phase

    Positive phase -> response leads reference.
    """
    x = np.asarray(reference, float)
    y = np.asarray(response, float)

    valid = np.isfinite(x) & np.isfinite(y)

    # Split into continuous valid sections
    edges = np.diff(
        np.r_[False, valid, False].astype(int)
    )
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]

    fs = 1.0 / (grid_minutes * 60.0)

    nperseg = int(
        round(welch_days * 86400 * fs)
    )
    noverlap = int(
        overlap_fraction * nperseg
    )
    step = nperseg - noverlap

    Sxx_sum = None
    Syy_sum = None
    Sxy_sum = None
    total_weight = 0
    frequency = None

    for i0, i1 in zip(starts, stops):
        xx = x[i0:i1]
        yy = y[i0:i1]

        if len(xx) < nperseg:
            continue

        nseg = 1 + (len(xx) - nperseg) // step

        kwargs = dict(
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="linear",
            scaling="density",
        )

        f, Sxx = welch(xx, **kwargs)
        _, Syy = welch(yy, **kwargs)
        _, Sxy = csd(xx, yy, **kwargs)

        if Sxx_sum is None:
            frequency = f
            Sxx_sum = nseg * Sxx
            Syy_sum = nseg * Syy
            Sxy_sum = nseg * Sxy
        else:
            Sxx_sum += nseg * Sxx
            Syy_sum += nseg * Syy
            Sxy_sum += nseg * Sxy

        total_weight += nseg

    if total_weight == 0:
        return None

    Sxx = Sxx_sum / total_weight
    Syy = Syy_sum / total_weight
    Sxy = Sxy_sum / total_weight

    coherence = (
        np.abs(Sxy) ** 2
        / (Sxx * Syy)
    )
    coherence = np.clip(coherence, 0, 1)

    phase = np.degrees(
        np.angle(Sxy)
    )

    return (
        frequency,
        coherence,
        phase,
    )


# ============================================================
# RUN + PLOT
# ============================================================

for frame_id in frames_to_plot:

    if frame_id not in matched:
        continue

    regular = make_10min_grid(
        matched[frame_id]
    )

    print(
        f"\n{frame_id}: "
        f"{len(regular)} samples on 10-min grid"
    )

    for reference_col, reference_label in reference_vars.items():

        fig, axes = plt.subplots(
            2, 1,
            figsize=(11, 7),
            sharex=True,
        )

        for response_col, response_label in response_vars.items():

            result = spectral_pair(
                regular[reference_col],
                regular[response_col],
            )

            if result is None:
                print(
                    f"No valid spectrum: "
                    f"{frame_id}, "
                    f"{response_col} vs {reference_col}"
                )
                continue

            f_hz, coh, phase = result

            f_cpd = (
                f_hz * 86400.0
            )

            use = (
                (f_cpd >= plot_fmin_cpd)
                & (f_cpd <= plot_fmax_cpd)
            )

            line, = axes[0].plot(
                f_cpd[use],
                coh[use],
                label=response_label,
            )

            axes[1].plot(
                f_cpd[use],
                phase[use],
                color=line.get_color(),
                label=response_label,
            )

        # Tidal constituent markers
            for f in [M2_cpd, S2_cpd]:
                ax.axvline(
                    f,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                )

        axes[0].set_ylabel(
            r"Coherence $\gamma^2$"
        )
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(
            ncol=3,
            fontsize=9,
        )

        axes[1].axhline(
            0,
            linestyle="--",
            linewidth=0.8,
        )
        axes[1].set_ylim(-180, 180)
        axes[1].set_ylabel(
            "Phase (deg)"
        )
        axes[1].set_xlabel(
            "Frequency (cycles/day)"
        )
        axes[1].grid(True, alpha=0.25)

        axes[1].set_xlim(
            plot_fmin_cpd,
            plot_fmax_cpd,
        )

        fig.suptitle(
            "Tidal-timescale coherence and phase\n"
            f"{frame_labels[frame_id]}: "
            f"responses relative to {reference_label}"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        plt.show()


# ============================================================
# FLOOD / EBB ASYMMETRY OF VELOCITY RMS
#
# Positive U_DA,along = EBB
# Negative U_DA,along = FLOOD
# ============================================================

for frame_id in frames_to_plot:

    if frame_id not in matched:
        continue

    d = matched[frame_id].loc[
        (matched[frame_id]["mid_time"] >= plot_start)
        & (matched[frame_id]["mid_time"] < plot_end)
    ].copy()

    # Current strength
    d["abs_along_current"] = np.abs(
        d[current_along_column]
    )

    # Flood / ebb classification
    d["tidal_direction"] = np.where(
        d[current_along_column] > 0,
        "Ebb",
        "Flood",
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        sharex=True,
    )

    variables = [
        (
            horizontal_ig_column,
            r"$U_{h,\mathrm{RMS,IG}}$ (m/s)",
        ),
        (
            horizontal_ss_column,
            r"$U_{h,\mathrm{RMS,SS}}$ (m/s)",
        ),
    ]

    for ax, (column, ylabel) in zip(
        axes,
        variables,
    ):

        for direction, marker in [
            ("Flood", "o"),
            ("Ebb", "^"),
        ]:

            x = d.loc[
                d["tidal_direction"] == direction,
                [
                    "abs_along_current",
                    column,
                ],
            ].dropna()

            ax.scatter(
                x["abs_along_current"],
                x[column],
                s=20,
                alpha=0.4,
                marker=marker,
                label=direction,
            )

            # Linear fit for each tidal direction
            if len(x) >= 3:

                slope, intercept = np.polyfit(
                    x["abs_along_current"],
                    x[column],
                    1,
                )

                xx = np.linspace(
                    x["abs_along_current"].min(),
                    x["abs_along_current"].max(),
                    100,
                )

                ax.plot(
                    xx,
                    slope * xx + intercept,
                    linewidth=2,
                )

                print(
                    f"{frame_id} | {column} | "
                    f"{direction}: "
                    f"slope = {slope:.4f}"
                )

        ax.set_xlabel(
            r"$|U_{\mathrm{DA,along}}|$ (m/s)"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

    fig.suptitle(
        "Flood–ebb asymmetry of velocity RMS response\n"
        f"{frame_labels[frame_id]}"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    plt.show()