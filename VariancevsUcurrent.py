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
# Velocity RMS from ADCP-HR/ADCP-UP analysis
# ------------------------------------------------------------

input_file_vel = (
    spectra_folder
    / "all_cases_ADCP_UP_velocity.csv"
)

# ------------------------------------------------------------
# Pressure-derived IG/SS wave statistics
#
# Water depth from this file is NOT used anymore.
# ------------------------------------------------------------

input_file_wave = (
    spectra_folder
    / "all_cases_wave_heights_ADCP_UP.csv"
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
    "30min"
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
# ADCP velocity RMS
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
    "hm0_ss_total_m"
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
# PRESSURE-DERIVED IG AND SS WAVE HEIGHTS — F1 VS F3
# ============================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 7),
    sharex=True,
)

wave_panels = [
    (hm0_ig_column, r"$H_{m0,\mathrm{IG}}$ (m)"),
    (hm0_ss_column, r"$H_{m0,\mathrm{SS}}$ (m)"),
]

colors = {
    "F1": "tab:blue",
    "F3": "tab:orange",
}

for frame_id in frames_to_plot:

    d = (
        df_wave.loc[
            (df_wave["frame_id"] == frame_id)
            & (df_wave["mid_time"] >= plot_start)
            & (df_wave["mid_time"] < plot_end)
        ]
        .sort_values("mid_time")
        .drop_duplicates(subset="mid_time")
    )

    if d.empty:
        continue

    for ax, (column, ylabel) in zip(axes, wave_panels):
        ax.plot(
            d["mid_time"],
            d[column],
            color=colors[frame_id],
            linewidth=1.0,
            label=frame_labels[frame_id],
        )

for ax, (_, ylabel) in zip(axes, wave_panels):
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)

axes[-1].set_xlabel("Time (UTC)", fontsize=15)
axes[-1].set_xlim(plot_start, plot_end)

fig.tight_layout()
plt.show()


# # ============================================================
# # FIGURES 1-2 — ORIGINAL HIGHEST-CELL TIME SERIES
# #
# # U_RMS,IG
# # U_RMS,SS
# # Ru
# # DEPTH-AVERAGED CURRENT SPEED
# # ADCP WATER DEPTH
# # ============================================================

# for frame_id in [
#     "F1",
#     "F3",
# ]:

#     if frame_id not in matched:
#         continue

#     d = (
#         matched[frame_id]
#         .loc[
#             (matched[frame_id]["mid_time"] >= plot_start)
#             & (matched[frame_id]["mid_time"] < plot_end)
#         ]
#         .sort_values("mid_time")
#         .copy()
#     )

#     columns_to_break = [
#         horizontal_ig_column,
#         horizontal_ss_column,
#         "Ru",
#         current_speed_column,
#         water_depth_column,
#     ]

#     # d = break_plot_gaps(
#     #     d,
#     #     columns_to_break,
#     # )

#     fig, axes = plt.subplots(
#         5,
#         1,
#         figsize=(13, 11),
#         sharex=True,
#     )

#     variables = [
#         (
#             horizontal_ig_column,
#             r"$U_{h,\mathrm{RMS,IG}}$ (m/s)",
#         ),
#         (
#             horizontal_ss_column,
#             r"$U_{h,\mathrm{RMS,SS}}$ (m/s)",
#         ),
#         (
#             "Ru",
#             r"$R_U="
#             r"U_{h,\mathrm{RMS,IG}}^2/"
#             r"U_{h,\mathrm{RMS,SS}}^2$",
#         ),
#         (
#             current_speed_column,
#             r"$|\mathbf{U}_{DA}|$ (m/s)",
#         ),
#         (
#             water_depth_column,
#             r"$h_{\mathrm{ADCP}}$ (m)",
#         ),
#     ]

#     for i, (ax, (column, ylabel)) in enumerate(zip(axes, variables)):
#         ax.plot(
#             d["mid_time"],
#             d[column],
#         )
#         ax.set_ylabel(ylabel)
#         ax.grid(True, which="both", alpha=0.3)

#         # Ru on logarithmic scale; other panels unchanged.
#         if column == "Ru":
#             ax.set_yscale("log")
#         elif i < 4:
#             ax.set_ylim(bottom=0)

#     axes[-1].set_xlabel("Time")

#     fig.suptitle(
#         "Velocity IG/SS modulation and "
#         "depth-averaged tidal current\n"
#         f"{frame_labels[frame_id]}, "
#         f"{plot_start:%Y-%m-%d} to "
#         f"{plot_end:%Y-%m-%d}"
#     )

#     fig.tight_layout(
#         rect=[0, 0, 1, 0.95]
#     )

#     plt.show()


# # ============================================================
# # FIGURE 3-4 — FOUR-CELL VELOCITY VARIANCE
# #
# # TOTAL HORIZONTAL VELOCITY VARIANCE, FOUR HR ADCP CELLS
# #   1. IG-band variance
# #   2. SS-band variance
# #   3. Ru = sigma_h,IG^2 / sigma_h,SS^2
# #   4. DEPTH-AVERAGED CURRENT SPEED |U_DA|
# #
# # Cell heights are read directly from the velocity statistics file.
# # The rest of the analyses below continue to use the original
# # highest-cell matched data and are unchanged.
# # ============================================================

# for frame_id in [
#     "F1",
#     "F3",
# ]:

#     if frame_id not in depth_averaged:
#         continue

#     # --------------------------------------------------------
#     # All available processed HR ADCP cells for this frame
#     # --------------------------------------------------------
#     d_cells = (
#         df_vel.loc[
#             (df_vel["frame_id"] == frame_id)
#             & (df_vel["mid_time"] >= plot_start)
#             & (df_vel["mid_time"] < plot_end)
#         ]
#         .copy()
#     )

#     if d_cells.empty:
#         continue

#     # Total horizontal velocity variances.
#     # horizontal_*_rms_m_s^2 = sigma_u^2 + sigma_v^2
#     d_cells["ig_horizontal_variance_m2_s2"] = (
#         d_cells[horizontal_ig_column] ** 2
#     )

#     d_cells["ss_horizontal_variance_m2_s2"] = (
#         d_cells[horizontal_ss_column] ** 2
#     )

#     d_cells["Ru_all_cells"] = (
#         d_cells["ig_horizontal_variance_m2_s2"]
#         / d_cells["ss_horizontal_variance_m2_s2"]
#     )

#     d_cells["Ru_all_cells"] = (
#         d_cells["Ru_all_cells"]
#         .replace([np.inf, -np.inf], np.nan)
#     )

#     # Cell number -> representative height above frame bottom.
#     # Use the median in case the same cell height is repeated
#     # across many burst rows.
#     cell_height_map = (
#         d_cells[[cell_column, height_column]]
#         .dropna()
#         .groupby(cell_column)[height_column]
#         .median()
#         .to_dict()
#     )

#     cells = sorted(
#         d_cells[cell_column]
#         .dropna()
#         .unique(),
#         key=lambda cell: cell_height_map.get(cell, np.nan),
#         reverse=True,
#     )

#     if len(cells) != 4:
#         print(
#             f"WARNING: {frame_id} has {len(cells)} processed cells "
#             f"in the velocity file: {cells}"
#         )

#     # --------------------------------------------------------
#     # Depth-averaged current speed
#     # --------------------------------------------------------
#     d_current = (
#         depth_averaged[frame_id]
#         .loc[
#             (depth_averaged[frame_id]["current_time"] >= plot_start)
#             & (depth_averaged[frame_id]["current_time"] < plot_end),
#             ["current_time", current_speed_column],
#         ]
#         .sort_values("current_time")
#         .copy()
#     )

#     # Break gaps in the current series without changing the data.
#     current_gap = (
#         d_current["current_time"]
#         .diff()
#         .dt.total_seconds()
#         > plot_gap_seconds
#     )
#     d_current.loc[current_gap, current_speed_column] = np.nan

#     # --------------------------------------------------------
#     # Plot
#     # --------------------------------------------------------
#     fig, axes = plt.subplots(
#         4,
#         1,
#         figsize=(13, 10),
#         sharex=True,
#     )

#     for cell in cells:

#         x = (
#             d_cells.loc[
#                 d_cells[cell_column] == cell,
#                 [
#                     "mid_time",
#                     "ig_horizontal_variance_m2_s2",
#                     "ss_horizontal_variance_m2_s2",
#                     "Ru_all_cells",
#                 ],
#             ]
#             .sort_values("mid_time")
#             .drop_duplicates("mid_time")
#             .copy()
#         )

#         # x = break_plot_gaps(
#         #     x,
#         #     [
#         #         "ig_horizontal_variance_m2_s2",
#         #         "ss_horizontal_variance_m2_s2",
#         #         "Ru_all_cells",
#         #     ],
#         # )

#         z = cell_height_map.get(cell, np.nan)

#         if np.isfinite(z):
#             label = (
#                 f"z={z:.2f} m"
#             )
#         else:
#             label = f"Cell {int(cell)}"

#         axes[0].plot(
#             x["mid_time"],
#             x["ig_horizontal_variance_m2_s2"],
#             linewidth=0.9,
#             label=label,
#         )

#         axes[1].plot(
#             x["mid_time"],
#             x["ss_horizontal_variance_m2_s2"],
#             linewidth=0.9,
#             label=label,
#         )

#         axes[2].plot(
#             x["mid_time"],
#             x["Ru_all_cells"],
#             linewidth=0.9,
#             label=label,
#         )

#     axes[3].plot(
#         d_current["current_time"],
#         d_current[current_speed_column],
#         linewidth=1.0,
#     )

#     axes[0].set_ylabel(
#         r"$\sigma_{h,IG}^{2}$ (m$^2$ s$^{-2}$)"
#     )

#     axes[1].set_ylabel(
#         r"$\sigma_{h,SS}^{2}$ (m$^2$ s$^{-2}$)"
#     )

#     axes[2].set_ylabel(
#         r"$R_U=\sigma_{h,IG}^{2}/\sigma_{h,SS}^{2}$"
#     )

#     axes[3].set_ylabel(
#         r"$|\mathbf{U}_{DA}|$ (m/s)"
#     )

#     axes[3].set_xlabel("Time")

#     # Velocity variances and Ru span orders of magnitude.
#     for ax in axes:
#         ax.grid(True, which="both", alpha=0.3)

#     for ax in axes[:3]:
#         ax.set_yscale("log")

#     axes[3].set_ylim(bottom=0)

#     # Heights are shown in the velocity-panel legends.
#     axes[0].legend(ncol=2, fontsize=9)
#     axes[1].legend(ncol=2, fontsize=9)
#     axes[2].legend(ncol=2, fontsize=9)

#     fig.suptitle(
#         "IG/SS horizontal velocity variance and "
#         "depth-averaged tidal current\n"
#         f"{frame_labels[frame_id]}, "
#         f"{plot_start:%Y-%m-%d} to "
#         f"{plot_end:%Y-%m-%d}"
#     )

#     fig.tight_layout(
#         rect=[0, 0, 1, 0.95]
#     )

#     plt.show()


# # ============================================================
# # FIGURES 5-6
# #
# # m0IG
# # m0SS
# # RH
# # DEPTH-AVERAGED CURRENT SPEED
# # ADCP WATER DEPTH
# # ============================================================

# for frame_id in [
#     "F1",
#     "F3",
# ]:

#     if frame_id not in matched:
#         continue


#     d = (
#         matched[
#             frame_id
#         ]
#         .loc[
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 >= plot_start
#             )
#             &
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 < plot_end
#             )
#         ]
#         .sort_values(
#             "mid_time"
#         )
#         .copy()
#     )


#     columns_to_break = [
#         "m0_IG",
#         "m0_SS",
#         "RH",
#         current_speed_column,
#         water_depth_column,
#     ]


#     # d = break_plot_gaps(
#     #     d,
#     #     columns_to_break,
#     # )


#     fig, axes = plt.subplots(
#         5,
#         1,
#         figsize=(
#             13,
#             11,
#         ),
#         sharex=True,
#     )


#     variables = [
#         (
#             "m0_IG",
#             r"$m_{0,\mathrm{IG}}$ "
#             r"(cm$^2$)",
#         ),

#         (
#             "m0_SS",
#             r"$m_{0,\mathrm{SS}}$ "
#             r"(cm$^2$)",
#         ),

#         (
#             "RH",
#             r"$R_H="
#             r"m_{0,\mathrm{IG}}/"
#             r"m_{0,\mathrm{SS}}$",
#         ),

#         (
#             current_speed_column,
#             r"$|\mathbf{U}_{DA}|$ (m/s)",
#         ),

#         (
#             water_depth_column,
#             r"$h_{\mathrm{ADCP}}$ (m)",
#         ),
#     ]


#     for i, (
#         ax,
#         (
#             column,
#             ylabel,
#         ),
#     ) in enumerate(
#         zip(
#             axes,
#             variables,
#         )
#     ):

#         ax.plot(
#             d["mid_time"],
#             d[column],
#         )

#         ax.set_ylabel(
#             ylabel
#         )

#         ax.grid(
#             True,
#             which="both",
#             alpha=0.3,
#         )


#         # m0 and RH panels
#         if i < 3:

#             ax.set_yscale(
#                 "log"
#             )


#         # Current speed
#         elif i == 3:

#             ax.set_ylim(
#                 bottom=0
#             )


#     axes[-1].set_xlabel(
#         "Time"
#     )


#     fig.suptitle(
#         "Surface-wave IG/SS modulation and "
#         "depth-averaged tidal current\n"
#         f"{frame_labels[frame_id]}, "
#         f"{plot_start:%Y-%m-%d} to "
#         f"{plot_end:%Y-%m-%d}"
#     )


#     fig.tight_layout(
#         rect=[
#             0,
#             0,
#             1,
#             0.95,
#         ]
#     )

#     plt.show()
    
# ============================================================
# FIGURES
#
# U_RMS,IG,cross& along
# U_RMS,SS,cross& along
# DEPTH-AVERAGED CURRENT SPEED
# ============================================================

# for frame_id in [
#     "F1",
#     "F3",
# ]:

#     if frame_id not in matched:
#         continue


#     d = (
#         matched[
#             frame_id
#         ]
#         .loc[
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 >= plot_start
#             )
#             &
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 < plot_end
#             )
#         ]
#         .sort_values(
#             "mid_time"
#         )
#         .copy()
#     )


#     columns_to_break = [
#         cross_ig_column,
#         along_ig_column,
#         cross_ss_column,
#         along_ss_column,
#         current_speed_column,
#     ]

#     # d = break_plot_gaps(
#     #     d,
#     #     columns_to_break,
#     # )


#     fig, axes = plt.subplots(
#         5,
#         1,
#         figsize=(
#             13,
#             11,
#         ),
#         sharex=True,
#     )


#     variables = [
#         (
#             cross_ig_column,
#             r"$U_{\mathrm{RMS,IG}}$ (m/s)",
#         ),

#         (
#             along_ss_column,
#             r"$V_{\mathrm{RMS,IG}}$ (m/s)",
#         ),

#         (
#             cross_ss_column,
#             r"$U_{\mathrm{RMS,SS}}$ (m/s)",
#         ),
#         (
#             along_ss_column,
#             r"$V_{\mathrm{RMS,SS}}$ (m/s)",
#         ),

#         (
#             current_speed_column,
#             r"$|\mathbf{U}_{DA}|$ (m/s)",
#         ),
#     ]


#     for (
#         ax,
#         (
#             column,
#             ylabel,
#         ),
#     ) in zip(
#         axes,
#         variables,
#     ):

#         ax.plot(
#             d["mid_time"],
#             d[column],
#         )

#         ax.set_ylabel(
#             ylabel
#         )

#         ax.grid(
#             True,
#             alpha=0.3,
#         )


#     # Nonnegative quantities
#     for ax in axes[:4]:

#         ax.set_ylim(
#             bottom=0
#         )


#     axes[-1].set_xlabel(
#         "Time"
#     )


#     fig.suptitle(
#         "Velocity IG/SS modulation and "
#         "depth-averaged tidal current\n"
#         f"{frame_labels[frame_id]}, "
#         f"{plot_start:%Y-%m-%d} to "
#         f"{plot_end:%Y-%m-%d}"
#     )


#     fig.tight_layout(
#         rect=[
#             0,
#             0,
#             1,
#             0.95,
#         ]
#     )

#     plt.show()


# ============================================================
# FIGURE
#
# Ru VS DEPTH-AVERAGED CURRENT SPEED
# RH VS DEPTH-AVERAGED CURRENT SPEED
# F1 / F3
# ============================================================

# fig, axes = plt.subplots(
#     2,
#     2,
#     figsize=(
#         10,
#         8,
#     ),
#     sharex="col",
# )


# for j, frame_id in enumerate(
#     [
#         "F1",
#         "F3",
#     ]
# ):

#     if frame_id not in matched:
#         continue


#     # Use exactly the same requested time window
#     # as the time-series figures.

#     d = (
#         matched[
#             frame_id
#         ]
#         .loc[
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 >= plot_start
#             )
#             &
#             (
#                 matched[
#                     frame_id
#                 ][
#                     "mid_time"
#                 ]
#                 < plot_end
#             )
#         ]
#         .copy()
#     )


#     # --------------------------------------------------------
#     # Ru versus current speed
#     # --------------------------------------------------------

#     x = (
#         d[
#             [
#                 current_speed_column,
#                 "Ru",
#             ]
#         ]
#         .dropna()
#     )


#     axes[
#         0,
#         j,
#     ].scatter(
#         x[
#             current_speed_column
#         ],
#         x[
#             "Ru"
#         ],
#         s=18,
#         alpha=0.5,
#     )


#     axes[
#         0,
#         j,
#     ].set_ylabel(
#         r"$R_U$"
#     )

#     axes[
#         0,
#         j,
#     ].set_title(
#         frame_labels[
#             frame_id
#         ]
#     )

#     axes[
#         0,
#         j,
#     ].set_ylim(
#         bottom=0
#     )

#     axes[
#         0,
#         j,
#     ].grid(
#         True,
#         alpha=0.3,
#     )


#     # --------------------------------------------------------
#     # RH versus current speed
#     # --------------------------------------------------------

#     x = (
#         d[
#             [
#                 current_speed_column,
#                 "RH",
#             ]
#         ]
#         .dropna()
#     )


#     axes[
#         1,
#         j,
#     ].scatter(
#         x[
#             current_speed_column
#         ],
#         x[
#             "RH"
#         ],
#         s=18,
#         alpha=0.5,
#     )


#     axes[
#         1,
#         j,
#     ].set_xlabel(
#         r"Depth-averaged current speed "
#         r"$|\mathbf{U}_{DA}|$ (m/s)"
#     )

#     axes[
#         1,
#         j,
#     ].set_ylabel(
#         r"$R_H$"
#     )

#     axes[
#         1,
#         j,
#     ].set_ylim(
#         bottom=0
#     )

#     axes[
#         1,
#         j,
#     ].grid(
#         True,
#         alpha=0.3,
#     )


# fig.suptitle(
#     "IG/SS ratios versus depth-averaged current speed\n"
#     f"{plot_start:%Y-%m-%d} to "
#     f"{plot_end:%Y-%m-%d}"
# )


# fig.tight_layout(
#     rect=[
#         0,
#         0,
#         1,
#         0.95,
#     ]
# )

# plt.show()

# ============================================================
# TIDAL-TIMESCALE COHERENCE + PHASE
# 4-cell velocity variance + surface-elevation variance
# relative to |U_DA|
# ============================================================

grid_minutes = 10
max_interp_gap = 6          # 6 × 10 min = 1 hour
welch_days = 5
overlap_fraction = 0.5

plot_fmin_cpd, plot_fmax_cpd = 0.5, 4.0
M2_cpd = 1.9323
M4_cpd = 2 * M2_cpd


def interp_short_gaps(s, max_gap):
    """Interpolate only internal gaps <= max_gap samples."""
    s = pd.Series(s, dtype=float).copy()
    miss = s.isna()

    if not miss.any():
        return s

    candidate = s.interpolate(method="linear", limit_area="inside")
    groups = miss.ne(miss.shift()).cumsum()

    for g in groups[miss].unique():
        idx = groups.index[(groups == g) & miss]

        if (
            len(idx) <= max_gap
            and idx[0] != s.index[0]
            and idx[-1] != s.index[-1]
            and np.isfinite(s.loc[idx[0] - 1])
            and np.isfinite(s.loc[idx[-1] + 1])
        ):
            s.loc[idx] = candidate.loc[idx]

    return s


def spectral_pair(reference, response):
    """
    Coherence and phase between reference and response.

    phase = response - reference
    positive phase -> response leads reference
    """
    x = np.asarray(reference, float)
    y = np.asarray(response, float)
    valid = np.isfinite(x) & np.isfinite(y)

    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]

    fs = 1.0 / (grid_minutes * 60.0)
    nperseg = int(round(welch_days * 86400 * fs))
    noverlap = int(overlap_fraction * nperseg)
    step = nperseg - noverlap

    Sxx_sum = Syy_sum = Sxy_sum = None
    total_weight = 0

    for i0, i1 in zip(starts, stops):

        xx = x[i0:i1]
        yy = y[i0:i1]

        if len(xx) < nperseg:
            continue

        nseg = 1 + (len(xx) - nperseg) // step

        kw = dict(
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="linear",
            scaling="density",
        )

        f, Sxx = welch(xx, **kw)
        _, Syy = welch(yy, **kw)
        _, Sxy = csd(xx, yy, **kw)

        if Sxx_sum is None:
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

    coherence = np.clip(
        np.abs(Sxy)**2 / (Sxx * Syy),
        0, 1,
    )

    phase = np.degrees(np.angle(Sxy))

    return f, coherence, phase


def regularize_series(time, values):
    """Put one time series on the common 10-min grid."""
    x = pd.DataFrame({
        "mid_time": time,
        "value": values,
    })

    x = (
        x.dropna(subset=["mid_time"])
        .sort_values("mid_time")
        .drop_duplicates("mid_time")
        .set_index("mid_time")
        .resample(f"{grid_minutes}min")
        .mean()
    )

    x["value"] = interp_short_gaps(
        x["value"].reset_index(drop=True),
        max_interp_gap,
    ).to_numpy()

    return x["value"]


# ============================================================
# RUN + PLOT
# ============================================================

for frame_id in frames_to_plot:

    if frame_id not in matched or frame_id not in depth_averaged:
        continue

    # --------------------------------------------------------
    # Common time grid
    # --------------------------------------------------------
    time_grid = pd.date_range(
        plot_start,
        plot_end,
        freq=f"{grid_minutes}min",
        inclusive="left",
    )

    # --------------------------------------------------------
    # Depth-averaged current speed
    # --------------------------------------------------------
    cur = depth_averaged[frame_id]

    UDA = regularize_series(
        cur["current_time"],
        cur[current_speed_column],
    ).reindex(time_grid)

    # --------------------------------------------------------
    # Surface-elevation variances
    # already calculated in matched[]
    # --------------------------------------------------------
    dm = matched[frame_id]

    m0 = {}

    for band in ["IG", "SS"]:
        col = f"m0_{band}"

        m0[band] = regularize_series(
            dm["mid_time"],
            dm[col],
        ).reindex(time_grid)

    # --------------------------------------------------------
    # Four-cell velocity data
    # df_vel is LONG FORMAT:
    # one row per burst × cell
    # --------------------------------------------------------
    dv = df_vel.loc[
        (df_vel["frame_id"] == frame_id)
        & (df_vel["mid_time"] >= plot_start)
        & (df_vel["mid_time"] < plot_end)
    ].copy()

    if dv.empty:
        continue

    # RMS² = total horizontal variance
    dv["var_IG"] = dv[horizontal_ig_column]**2
    dv["var_SS"] = dv[horizontal_ss_column]**2

    # Representative height for each selected cell
    cell_height_map = (
        dv.groupby(cell_column)[height_column]
        .median()
        .to_dict()
    )

    frame_cells = sorted(
        dv[cell_column].dropna().unique(),
        reverse=True,
    )

    print(
        f"\n{frame_id}: coherence analysis using "
        f"cells {frame_cells}"
    )

    # --------------------------------------------------------
    # 2 × 2 figure
    #
    # columns: IG / SS
    # rows:    coherence / phase
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 8),
        sharex=True,
        sharey="row",
    )

    for j, band in enumerate(["IG", "SS"]):

        ax_coh = axes[0, j]
        ax_phase = axes[1, j]

        # ----------------------------------------------------
        # Velocity variance — each cell
        # ----------------------------------------------------
        for cell in frame_cells:

            dc = dv.loc[
                dv[cell_column] == cell
            ].sort_values("mid_time")

            variance = regularize_series(
                dc["mid_time"],
                dc[f"var_{band}"],
            ).reindex(time_grid)

            result = spectral_pair(
                UDA,
                variance,
            )

            if result is None:
                print(
                    f"No valid spectrum: "
                    f"{frame_id}, {band}, cell {cell}"
                )
                continue

            f_hz, coh, phase = result
            f_cpd = f_hz * 86400.0

            use = (
                (f_cpd >= plot_fmin_cpd)
                & (f_cpd <= plot_fmax_cpd)
            )

            z = cell_height_map.get(cell, np.nan)

            line, = ax_coh.plot(
                f_cpd[use],
                coh[use],
                label=rf"$\sigma_h^2$, z={z:.2f} m",
            )

            ax_phase.plot(
                f_cpd[use],
                phase[use],
                color=line.get_color(),
            )

        # ----------------------------------------------------
        # Surface-elevation variance
        # ----------------------------------------------------
        result = spectral_pair(
            UDA,
            m0[band],
        )

        if result is not None:

            f_hz, coh, phase = result
            f_cpd = f_hz * 86400.0

            use = (
                (f_cpd >= plot_fmin_cpd)
                & (f_cpd <= plot_fmax_cpd)
            )

            ax_coh.plot(
                f_cpd[use],
                coh[use],
                "k--",
                linewidth=2,
                label=r"$m_{0,\eta}$",
            )

            ax_phase.plot(
                f_cpd[use],
                phase[use],
                "k--",
                linewidth=2,
            )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------
        for ax in [ax_coh, ax_phase]:

            ax.axvline(
                M2_cpd,
                color="0.6",
                linestyle=":",
                linewidth=0.8,
            )

            ax.axvline(
                M4_cpd,
                color="0.4",
                linestyle="--",
                linewidth=0.9,
            )

            ax.set_xlim(
                plot_fmin_cpd,
                plot_fmax_cpd,
            )

            ax.grid(True, alpha=0.25)

        ax_coh.set_title(f"{band} band")
        ax_coh.set_ylim(0, 1)

        ax_phase.axhline(
            0,
            color="0.5",
            linestyle=":",
            linewidth=0.8,
        )

        ax_phase.set_ylim(-180, 180)
        ax_phase.set_yticks(
            [-180, -90, 0, 90, 180]
        )

        ax_phase.set_xlabel(
            "Frequency (cycles/day)"
        )

    axes[0, 0].set_ylabel(
        r"Coherence $\gamma^2$"
    )

    axes[1, 0].set_ylabel(
        "Phase (deg)"
    )

    axes[0, 0].legend(
        fontsize=9,
        ncol=2,
    )

    fig.suptitle(
        "Tidal-timescale coherence and phase\n"
        f"{frame_labels[frame_id]}: variance relative to "
        r"$|\mathbf{U}_{DA}|$"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    plt.show()

# ============================================================
# VERTICAL STRUCTURE OF IG RMS - CURRENT DEPENDENCE
#
# One figure per frame
# Four panels = four measurement heights
#
# x = total depth-averaged current speed |U_DA|
# y = horizontal IG RMS
#
# Flood / ebb:
#   alongshore U_DA < 0 -> Flood
#   alongshore U_DA > 0 -> Ebb
#
# Raw bursts = faint scatter
# Overlay    = binned mean ± standard deviation
# ============================================================

speed_bin_width = 0.10       # m/s
minimum_bin_count = 3        # minimum bursts required per bin


for frame_id in frames_to_plot:

    if frame_id not in depth_averaged:
        continue


    # --------------------------------------------------------
    # ALL FOUR velocity cells
    # --------------------------------------------------------

    dv = (
        df_vel.loc[
            (df_vel["frame_id"] == frame_id)
            & (df_vel["mid_time"] >= plot_start)
            & (df_vel["mid_time"] < plot_end)
        ]
        .copy()
    )


    if dv.empty:
        continue


    # --------------------------------------------------------
    # Representative height of each cell
    # --------------------------------------------------------

    cell_height_map = (
        dv.groupby(cell_column)[height_column]
        .median()
        .to_dict()
    )


    # Highest -> lowest
    cells = sorted(
        dv[cell_column]
        .dropna()
        .unique(),
        key=lambda cell:
            cell_height_map.get(
                cell,
                np.nan,
            ),
        reverse=True,
    )


    # --------------------------------------------------------
    # Depth-averaged current
    # --------------------------------------------------------

    current = (
        depth_averaged[frame_id]
        .copy()
        .sort_values("current_time")
    )

    # --------------------------------------------------------
    # Match current to EVERY velocity cell
    # --------------------------------------------------------

    dv = (
        dv
        .sort_values("mid_time")
    )


    dv = pd.merge_asof(
        dv,
        current[
            [
                "current_time",
                current_speed_column,
                current_along_column,
            ]
        ],
        left_on="mid_time",
        right_on="current_time",
        direction="nearest",
        tolerance=current_merge_tolerance,
    )


    # --------------------------------------------------------
    # Current magnitude
    # --------------------------------------------------------

    # Magnitude of alongshore depth-averaged current
    dv["abs_V_DA"] = np.abs(
        dv[current_along_column]
    )


    # --------------------------------------------------------
    # Flood / ebb classification
    # --------------------------------------------------------

    dv["tidal_direction"] = np.where(
        dv[current_along_column] > 0,
        "Ebb",
        "Flood",
    )


    # Remove rows without the variables needed here
    dv = (
        dv.dropna(
            subset=[
                "abs_V_DA",
                current_along_column,
                horizontal_ig_column,
            ]
        )
        .copy()
    )


    if dv.empty:
        continue


    # ========================================================
    # COMMON CURRENT-SPEED BINS
    #
    # Same bins for:
    #   - every height
    #   - flood
    #   - ebb
    #
    # This is important for direct vertical comparison.
    # ========================================================

    max_speed = float(
        np.nanmax(
            dv["abs_V_DA"]
        )
    )


    speed_edges = np.arange(
        0.0,
        max_speed + speed_bin_width,
        speed_bin_width,
    )


    # Make sure the last observed value is inside the bins
    if speed_edges[-1] <= max_speed:

        speed_edges = np.append(
            speed_edges,
            speed_edges[-1] + speed_bin_width,
        )


    speed_centers = (
        speed_edges[:-1]
        + speed_edges[1:]
    ) / 2.0


    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(
        1,
        len(cells),
        figsize=(
            4.0 * len(cells),
            4.8,
        ),
        sharex=True,
        sharey=True,
    )


    axes = np.atleast_1d(
        axes
    )


    # --------------------------------------------------------
    # Each panel = one height
    # --------------------------------------------------------

    for ax, cell in zip(
        axes,
        cells,
    ):

        dc = (
            dv.loc[
                dv[cell_column] == cell
            ]
            .copy()
        )


        height_m = (
            cell_height_map.get(
                cell,
                np.nan,
            )
        )


        # ====================================================
        # Flood / ebb separately
        # ====================================================

        for direction, marker, color in [
            ("Flood", "o", "tab:blue"),
            ("Ebb", "^", "tab:orange"),
        ]:

            x = (
                dc.loc[
                    dc["tidal_direction"]
                    == direction,
                    [
                        "abs_V_DA",
                        horizontal_ig_column,
                    ],
                ]
                .dropna()
                .copy()
            )


            if x.empty:
                continue


            # ------------------------------------------------
            # Raw burst scatter
            # ------------------------------------------------

            ax.scatter(
                x["abs_V_DA"],
                x[horizontal_ig_column],
                s=15,
                alpha=0.15,          # faint raw observations
                marker=marker,
                color=color,
                label=f"{direction} bursts",
            )

            # ------------------------------------------------
            # Assign current-speed bins
            # ------------------------------------------------

            x["speed_bin"] = pd.cut(
                x["abs_V_DA"],
                bins=speed_edges,
                labels=False,
                include_lowest=True,
                right=False,
            )


            # ------------------------------------------------
            # Mean ± standard deviation in each bin
            # ------------------------------------------------

            stats = (
                x.groupby(
                    "speed_bin",
                    observed=True,
                )[horizontal_ig_column]
                .agg(
                    mean="mean",
                    std="std",
                    n="count",
                )
                .reset_index()
            )


            # Require sufficient observations
            stats = (
                stats.loc[
                    stats["n"]
                    >= minimum_bin_count
                ]
                .copy()
            )


            if stats.empty:
                continue


            # Actual bin centres
            stats["speed_center"] = (
                stats["speed_bin"]
                .astype(int)
                .map(
                    lambda i:
                        speed_centers[i]
                )
            )


            # ------------------------------------------------
            # Overlay mean ± standard deviation
            # ------------------------------------------------

            ax.errorbar(
                stats["speed_center"],
                stats["mean"],
                yerr=stats["std"],
                fmt=marker + "-",
                color=color,
                alpha=1.0,           # fully opaque
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linewidth=2.5,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                label=f"{direction} mean ± SD",
                zorder=10,
            )

        # ====================================================
        # PANEL FORMATTING
        # ====================================================

        ax.set_title(
            f"z = {height_m:.2f} m"
        )


        ax.set_xlabel(
            r"$|\mathbf{V}_{DA}|$ (m/s)"
        )


        ax.set_xlim(
            left=0
        )


        ax.set_ylim(
            bottom=0
        )


        ax.grid(
            True,
            alpha=0.3,
        )


    # --------------------------------------------------------
    # Shared y label
    # --------------------------------------------------------

    axes[0].set_ylabel(
        r"$U_{h,\mathrm{RMS,IG}}$ (m/s)"
    )


    # --------------------------------------------------------
    # Legend
    #
    # One legend only because all panels use same meaning
    # --------------------------------------------------------

    axes[0].legend(
        fontsize=8
    )


    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    fig.suptitle(
        "Vertical structure of IG velocity response "
        "to depth-averaged current\n"
        f"{frame_labels[frame_id]} — "
        "Flood vs Ebb; "
        f"mean ± SD in "
        f"{speed_bin_width:.2f} m/s bins"
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )


    plt.show()
    
# ============================================================
# RESIDUAL IG RMS VERSUS |V_DA|
#
# Residual IG velocity:
#
#   S_res(f) = max[S_measured(f) - S_LWT(f), 0]
#
#   U_RMS,IG,res = sqrt( integral_IG S_res(f) df )
#
# One figure per frame.
# One panel per measurement height.
#
# x = |V_DA| = magnitude of depth-averaged alongshore current
# y = residual horizontal IG RMS
#
# Flood / ebb:
#   V_DA < 0 -> Flood
#   V_DA > 0 -> Ebb
#
# Raw bursts = faint scatter
# Overlay    = binned mean ± standard deviation
# ============================================================


# # ------------------------------------------------------------
# # READ RESIDUAL RMS FILE
# # ------------------------------------------------------------

# residual_rms = pd.read_parquet(
#     spectra_folder
#     / "IG_residual_RMS_all_cells.parquet"
# )


# print(
#     "\nResidual RMS file:"
# )

# print(
#     residual_rms.columns.tolist()
# )


# # ------------------------------------------------------------
# # Normalize time
# # ------------------------------------------------------------

# residual_rms["mid_time"] = (
#     normalize_time(
#         residual_rms["mid_time"]
#     )
# )


# # ------------------------------------------------------------
# # Column containing residual RMS
# # ------------------------------------------------------------

# residual_rms_column = (
#     "urms_ig_res_m_s"
# )


# # ============================================================
# # PLOT
# # ============================================================

# for frame_id in frames_to_plot:

#     if frame_id not in depth_averaged:
#         continue


#     # --------------------------------------------------------
#     # Residual RMS for this frame and deployment period
#     # --------------------------------------------------------

#     dr = (
#         residual_rms.loc[
#             (
#                 residual_rms["frame_id"]
#                 == frame_id
#             )
#             & (
#                 residual_rms["mid_time"]
#                 >= plot_start
#             )
#             & (
#                 residual_rms["mid_time"]
#                 < plot_end
#             )
#         ]
#         .copy()
#     )


#     if dr.empty:

#         print(
#             f"No residual RMS data for {frame_id}"
#         )

#         continue


#     # --------------------------------------------------------
#     # Representative height for each instrument/cell
#     #
#     # IMPORTANT:
#     # Cell numbers can overlap between ADCP-UP and ADCP-HR,
#     # so identify a measurement level by instrument + cell.
#     # --------------------------------------------------------

#     height_table = (
#         dr[
#             [
#                 "instrument",
#                 "cell_number",
#                 "height_m",
#             ]
#         ]
#         .dropna()
#         .groupby(
#             [
#                 "instrument",
#                 "cell_number",
#             ],
#             as_index=False,
#         )["height_m"]
#         .median()
#         .sort_values(
#             "height_m",
#             ascending=False,
#         )
#         .reset_index(drop=True)
#     )


#     # --------------------------------------------------------
#     # Match depth-averaged current to residual RMS bursts
#     # --------------------------------------------------------

#     current = (
#         depth_averaged[frame_id]
#         .copy()
#         .sort_values(
#             "current_time"
#         )
#     )


#     dr = (
#         dr.sort_values(
#             "mid_time"
#         )
#     )


#     dr = pd.merge_asof(
#         dr,
#         current[
#             [
#                 "current_time",
#                 current_along_column,
#             ]
#         ],
#         left_on="mid_time",
#         right_on="current_time",
#         direction="nearest",
#         tolerance=current_merge_tolerance,
#     )


#     # --------------------------------------------------------
#     # |V_DA|
#     # --------------------------------------------------------

#     dr["abs_V_DA"] = np.abs(
#         dr[current_along_column]
#     )


#     # --------------------------------------------------------
#     # Flood / ebb
#     # --------------------------------------------------------

#     dr["tidal_direction"] = np.where(
#         dr[current_along_column] > 0,
#         "Ebb",
#         "Flood",
#     )


#     # --------------------------------------------------------
#     # Remove invalid observations
#     # --------------------------------------------------------

#     dr = (
#         dr.dropna(
#             subset=[
#                 "abs_V_DA",
#                 current_along_column,
#                 residual_rms_column,
#             ]
#         )
#         .copy()
#     )


#     if dr.empty:
#         continue


#     # ========================================================
#     # COMMON |V_DA| BINS
#     #
#     # Same bins at all eight heights and for flood/ebb.
#     # ========================================================

#     max_speed = float(
#         np.nanmax(
#             dr["abs_V_DA"]
#         )
#     )


#     speed_edges = np.arange(
#         0.0,
#         max_speed + speed_bin_width,
#         speed_bin_width,
#     )


#     if speed_edges[-1] <= max_speed:

#         speed_edges = np.append(
#             speed_edges,
#             speed_edges[-1]
#             + speed_bin_width,
#         )


#     speed_centers = (
#         speed_edges[:-1]
#         + speed_edges[1:]
#     ) / 2.0


#     # ========================================================
#     # FIGURE
#     #
#     # 8 panels:
#     # ADCP-UP + ADCP-HR ordered by physical height.
#     # ========================================================

#     n_heights = len(
#         height_table
#     )


#     fig, axes = plt.subplots(
#         2,
#         4,
#         figsize=(16, 9),
#         sharex=True,
#         sharey=True,
#     )


#     axes = (
#         np.asarray(axes)
#         .ravel()
#     )


#     # --------------------------------------------------------
#     # One panel per physical measurement height
#     # --------------------------------------------------------

#     for ax, (_, hrow) in zip(
#         axes,
#         height_table.iterrows(),
#     ):

#         instrument = (
#             hrow["instrument"]
#         )

#         cell = int(
#             hrow["cell_number"]
#         )

#         height_m = float(
#             hrow["height_m"]
#         )


#         dc = (
#             dr.loc[
#                 (
#                     dr["instrument"]
#                     == instrument
#                 )
#                 & (
#                     dr["cell_number"]
#                     == cell
#                 )
#             ]
#             .copy()
#         )


#         # ====================================================
#         # FLOOD / EBB
#         # ====================================================

#         for direction, marker, color in [
#             (
#                 "Flood",
#                 "o",
#                 "tab:blue",
#             ),
#             (
#                 "Ebb",
#                 "^",
#                 "tab:orange",
#             ),
#         ]:

#             x = (
#                 dc.loc[
#                     dc[
#                         "tidal_direction"
#                     ]
#                     == direction,
#                     [
#                         "abs_V_DA",
#                         residual_rms_column,
#                     ],
#                 ]
#                 .dropna()
#                 .copy()
#             )


#             if x.empty:
#                 continue


#             # ------------------------------------------------
#             # Raw bursts
#             # ------------------------------------------------

#             ax.scatter(
#                 x["abs_V_DA"],
#                 x[
#                     residual_rms_column
#                 ],
#                 s=15,
#                 alpha=0.15,
#                 marker=marker,
#                 color=color,
#                 label=(
#                     f"{direction} bursts"
#                 ),
#             )


#             # ------------------------------------------------
#             # Current-speed bins
#             # ------------------------------------------------

#             x["speed_bin"] = pd.cut(
#                 x["abs_V_DA"],
#                 bins=speed_edges,
#                 labels=False,
#                 include_lowest=True,
#                 right=False,
#             )


#             # ------------------------------------------------
#             # Mean ± standard deviation
#             # ------------------------------------------------

#             stats = (
#                 x.groupby(
#                     "speed_bin",
#                     observed=True,
#                 )[
#                     residual_rms_column
#                 ]
#                 .agg(
#                     mean="mean",
#                     std="std",
#                     n="count",
#                 )
#                 .reset_index()
#             )


#             stats = (
#                 stats.loc[
#                     stats["n"]
#                     >= minimum_bin_count
#                 ]
#                 .copy()
#             )


#             if stats.empty:
#                 continue


#             stats[
#                 "speed_center"
#             ] = (
#                 stats["speed_bin"]
#                 .astype(int)
#                 .map(
#                     lambda i:
#                         speed_centers[i]
#                 )
#             )


#             # ------------------------------------------------
#             # Binned mean ± standard deviation
#             # ------------------------------------------------

#             ax.errorbar(
#                 stats["speed_center"],
#                 stats["mean"],
#                 yerr=stats["std"],
#                 fmt=marker + "-",
#                 color=color,
#                 alpha=1.0,
#                 markersize=8,
#                 markeredgecolor="black",
#                 markeredgewidth=0.7,
#                 linewidth=2.5,
#                 capsize=4,
#                 elinewidth=1.8,
#                 label=(
#                     f"{direction} "
#                     r"mean $\pm$ std"
#                 ),
#             )


#         # ----------------------------------------------------
#         # Panel formatting
#         # ----------------------------------------------------

#         ax.set_title(
#             f"z = {height_m:.2f} m\n"
#             f"{instrument}"
#         )

#         ax.set_xlim(
#             left=0
#         )

#         ax.set_ylim(
#             bottom=0
#         )

#         ax.grid(
#             True,
#             alpha=0.25,
#         )


#     # --------------------------------------------------------
#     # Hide unused panels if fewer than 8 levels exist
#     # --------------------------------------------------------

#     for ax in axes[
#         n_heights:
#     ]:

#         ax.set_visible(
#             False
#         )


#     # --------------------------------------------------------
#     # Common labels
#     # --------------------------------------------------------

#     fig.supxlabel(
#         r"$|V_{DA}|$ (m/s)"
#     )


#     fig.supylabel(
#         r"$U_{\mathrm{RMS,IG,res}}$ (m/s)"
#     )


#     # One legend only
#     axes[0].legend(
#         fontsize=8,
#         loc="best",
#     )


#     fig.suptitle(
#         f"{frame_labels[frame_id]}: "
#         "residual IG velocity versus "
#         "alongshore depth-averaged current\n"
#         r"$S_{\mathrm{res}}="
#         r"\max(S_{\mathrm{measured}}"
#         r"-S_{\mathrm{LWT}},0)$"
#     )


#     fig.tight_layout(
#         rect=[
#             0.03,
#             0.03,
#             1,
#             0.93,
#         ]
#     )


#     plt.show()