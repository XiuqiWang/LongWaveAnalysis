# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:35:02 2026

@author: WangX3
"""

# -*- coding: utf-8 -*-
"""
Read processed ADCP depth-averaged mean velocity.

Workflow
--------
1. Read:
       velocity_dabmean
       water_depth
       time

2. Decode scale factors.

3. Plot the original geographic components:
       U_E = Eastward depth-averaged velocity
       U_N = Northward depth-averaged velocity
       h   = water depth

   This figure can be compared directly with the KG2.0 report.

4. Rotate U_E and U_N to:
       cross_shore
       alongshore

   Positive cross-shore:
       nearshore -> offshore

   Positive alongshore:
       90 degrees counterclockwise from positive cross-shore.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

file_path = Path(
    r"C:\dev\Python\LongWaveAnalysis\ADCP"
    r"\UP\adcp_dvn_201804_f1p1_000_da10.nc"
)

# Coordinate definition used in the existing analysis
nearshore_point = (
    52.23317,
    4.3873215,
)  # DVN3

offshore_point = (
    52.28087,
    4.2432895,
)  # DVN1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def scalar_attribute(
    dataset,
    name,
    default,
):
    """Read a scalar HDF5 attribute."""

    value = dataset.attrs.get(
        name,
        default,
    )

    return float(
        np.asarray(value).squeeze()
    )


def read_scaled_variable(
    dataset,
):
    """
    Read a complete HDF5 variable and apply
    scale_factor/add_offset.

    Explicit fill values are converted to NaN.
    """

    raw = np.asarray(
        dataset[...],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Fill values
    # --------------------------------------------------------

    invalid = np.zeros(
        raw.shape,
        dtype=bool,
    )

    for attribute_name in [
        "_FillValue",
        "missing_value",
    ]:

        if attribute_name not in dataset.attrs:
            continue

        fill_values = np.asarray(
            dataset.attrs[attribute_name]
        ).ravel()

        for fill_value in fill_values:
            invalid |= (
                raw == float(fill_value)
            )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scale = scalar_attribute(
        dataset,
        "scale_factor",
        1.0,
    )

    offset = scalar_attribute(
        dataset,
        "add_offset",
        0.0,
    )

    values = (
        raw * scale
        + offset
    )

    values[invalid] = np.nan

    return values


def rotate_enu_to_cross_along(
    east,
    north,
    point_from,
    point_to,
):
    """
    Rotate East/North velocity into cross-shore/alongshore.

    point_from -> point_to defines positive cross-shore.

    Positive alongshore is 90 degrees counterclockwise
    from positive cross-shore.
    """

    east = np.asarray(
        east,
        dtype=float,
    )

    north = np.asarray(
        north,
        dtype=float,
    )

    if east.shape != north.shape:
        raise ValueError(
            "east and north must have the same shape."
        )

    lat1, lon1 = map(
        float,
        point_from,
    )

    lat2, lon2 = map(
        float,
        point_to,
    )

    # --------------------------------------------------------
    # Local geographic -> East/North displacement
    # --------------------------------------------------------

    earth_radius_m = 6_371_000.0

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)

    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)

    mean_lat_rad = (
        0.5
        * (
            lat1_rad
            + lat2_rad
        )
    )

    delta_east_m = (
        earth_radius_m
        * (
            lon2_rad
            - lon1_rad
        )
        * np.cos(mean_lat_rad)
    )

    delta_north_m = (
        earth_radius_m
        * (
            lat2_rad
            - lat1_rad
        )
    )

    transect_length_m = np.hypot(
        delta_east_m,
        delta_north_m,
    )

    if transect_length_m == 0:
        raise ValueError(
            "The two transect points are identical."
        )

    # --------------------------------------------------------
    # Cross-shore unit vector
    # --------------------------------------------------------

    cross_east = (
        delta_east_m
        / transect_length_m
    )

    cross_north = (
        delta_north_m
        / transect_length_m
    )

    # --------------------------------------------------------
    # Alongshore unit vector
    # 90 deg counterclockwise from cross-shore
    # --------------------------------------------------------

    along_east = -cross_north
    along_north = cross_east

    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    cross_shore = (
        east * cross_east
        + north * cross_north
    )

    alongshore = (
        east * along_east
        + north * along_north
    )

    # Bearing clockwise from North
    bearing_deg = (
        np.degrees(
            np.arctan2(
                delta_east_m,
                delta_north_m,
            )
        )
        + 360.0
    ) % 360.0

    metadata = {
        "positive_cross_shore_bearing_deg":
            bearing_deg,

        "transect_length_m":
            transect_length_m,

        "cross_shore_unit_east":
            cross_east,

        "cross_shore_unit_north":
            cross_north,

        "alongshore_unit_east":
            along_east,

        "alongshore_unit_north":
            along_north,
    }

    return (
        cross_shore,
        alongshore,
        metadata,
    )


# ============================================================
# READ FILE
# ============================================================

print("Reading:")
print(file_path)

with h5py.File(
    file_path,
    "r",
) as f:

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    time_raw = read_scaled_variable(
        f["time"]
    )

    time = pd.to_datetime(
        time_raw,
        unit="s",
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Depth-averaged mean velocity
    #
    # Shape = (3, time)
    #
    # component 0 = East
    # component 1 = North
    # component 2 = Up
    # --------------------------------------------------------

    velocity_dabmean = read_scaled_variable(
        f["velocity_dabmean"]
    )

    if velocity_dabmean.shape[0] != 3:
        raise ValueError(
            "Expected velocity_dabmean shape "
            "(3, time), but obtained "
            f"{velocity_dabmean.shape}"
        )

    velocity_east = (
        velocity_dabmean[0, :]
    )

    velocity_north = (
        velocity_dabmean[1, :]
    )

    velocity_up = (
        velocity_dabmean[2, :]
    )

    # --------------------------------------------------------
    # Water depth
    # --------------------------------------------------------

    water_depth = read_scaled_variable(
        f["water_depth"]
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(
    {
        "time":
            time,

        "depth_avg_east_m_s":
            velocity_east,

        "depth_avg_north_m_s":
            velocity_north,

        "depth_avg_up_m_s":
            velocity_up,

        "water_depth_m":
            water_depth,
    }
)

df = (
    df
    .dropna(
        subset=["time"]
    )
    .sort_values("time")
    .reset_index(drop=True)
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nData summary")

print(
    "Start:",
    df["time"].iloc[0],
)

print(
    "End:  ",
    df["time"].iloc[-1],
)

print(
    "Rows: ",
    len(df),
)

dt = (
    df["time"]
    .diff()
    .dt.total_seconds()
)

print(
    "Median time interval:",
    dt.median(),
    "s",
)

print(
    "Median time interval:",
    dt.median() / 60.0,
    "min",
)

print("\nMissing values:")
print(
    df[
        [
            "depth_avg_east_m_s",
            "depth_avg_north_m_s",
            "water_depth_m",
        ]
    ]
    .isna()
    .sum()
)

# ============================================================
# IDENTIFY NO-MEASUREMENT RECORDS
# ============================================================

# Zero water depth is physically impossible here and therefore
# identifies the regularly occurring no-measurement periods.
missing = (
    ~np.isfinite(df["water_depth_m"])
    | (df["water_depth_m"] <= 0)
)

print(
    "Records identified as no measurement:",
    missing.sum(),
)

print(
    "Fraction missing:",
    missing.mean(),
)


# Keep original data untouched
df_plot = df.copy()

columns_to_mask = [
    "depth_avg_east_m_s",
    "depth_avg_north_m_s",
    "depth_avg_up_m_s",
    "water_depth_m",
]

df_plot.loc[
    missing,
    columns_to_mask,
] = np.nan

# ============================================================
# CHECK GAP LENGTHS
# ============================================================

missing_groups = (
    missing.ne(missing.shift())
    .cumsum()
)

gap_lengths = (
    df.loc[missing]
    .groupby(missing_groups[missing])
    .size()
)

print("\nMissing-gap lengths in samples:")
print(gap_lengths.value_counts().sort_index())

dt_seconds = (
    df["time"]
    .diff()
    .dt.total_seconds()
    .median()
)

print(
    "\nMedian sampling interval:",
    dt_seconds / 60,
    "minutes",
)

print("\nGap durations:")
for n, count in gap_lengths.value_counts().sort_index().items():
    print(
        f"{n} samples = "
        f"{n * dt_seconds / 60:.1f} min "
        f"({count} occurrences)"
    )

# ============================================================
# TIME-BASED INTERPOLATION
# ============================================================
interp_columns = [
    "depth_avg_east_m_s",
    "depth_avg_north_m_s",
    "depth_avg_up_m_s",
    "water_depth_m",
]

df_plot = (
    df_plot
    .set_index("time")
)

for col in interp_columns:

    df_plot[col] = (
        df_plot[col]
        .interpolate(
            method="time",
            limit=3,
            limit_area="inside",
        )
    )

df_plot = (
    df_plot
    .reset_index()
)

# ============================================================
# REPORT-STYLE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 7),
    sharex=True,
)

axes[0].plot(
    df_plot["time"],
    df_plot["depth_avg_east_m_s"],
    linewidth=0.8,
    label=r"$U_E$",
)

axes[0].plot(
    df_plot["time"],
    df_plot["depth_avg_north_m_s"],
    linewidth=0.8,
    label=r"$U_N$",
)

axes[0].axhline(
    0,
    linewidth=0.6,
    linestyle="--",
)

axes[0].set_ylabel(
    "Depth-averaged velocity (m/s)"
)

axes[0].legend()
axes[0].grid(True, alpha=0.25)


axes[1].plot(
    df_plot["time"],
    df_plot["water_depth_m"],
    linewidth=0.8,
)

axes[1].set_ylabel(
    "Water depth (m)"
)

axes[1].set_xlabel(
    "Time"
)

axes[1].grid(True, alpha=0.25)


fig.suptitle(
    "ADCP depth-averaged velocity and water depth\n"
    "Short no-measurement intervals interpolated"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.show()


# ============================================================
# ROTATE EAST/NORTH TO CROSS-SHORE / ALONGSHORE
# ============================================================

(
    df["depth_avg_cross_shore_m_s"],
    df["depth_avg_alongshore_m_s"],
    rotation_info,
) = rotate_enu_to_cross_along(
    east=df["depth_avg_east_m_s"],
    north=df["depth_avg_north_m_s"],
    point_from=nearshore_point,
    point_to=offshore_point,
)


# Current-vector magnitude
df["depth_avg_current_speed_m_s"] = np.hypot(
    df["depth_avg_cross_shore_m_s"],
    df["depth_avg_alongshore_m_s"],
)


# ============================================================
# ROTATION INFORMATION
# ============================================================

print("\nRotation information")

for key, value in rotation_info.items():
    print(
        f"{key}: {value}"
    )


# Verify rotation preserves horizontal-vector magnitude
speed_enu = np.hypot(
    df["depth_avg_east_m_s"],
    df["depth_avg_north_m_s"],
)

speed_rotated = np.hypot(
    df["depth_avg_cross_shore_m_s"],
    df["depth_avg_alongshore_m_s"],
)

rotation_error = np.nanmax(
    np.abs(
        speed_enu
        - speed_rotated
    )
)

print(
    "\nMaximum speed difference "
    "before/after rotation:",
    rotation_error,
    "m/s",
)


# ============================================================
# FIGURE 2
# ROTATED CURRENT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 8),
    sharex=True,
)

# Cross-shore
axes[0].plot(
    df["time"],
    df["depth_avg_cross_shore_m_s"],
)

axes[0].axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)

axes[0].set_ylabel(
    r"$U_{\mathrm{cross}}$"
    "\n(m/s)"
)

axes[0].grid(
    True,
    alpha=0.3,
)


# Alongshore
axes[1].plot(
    df["time"],
    df["depth_avg_alongshore_m_s"],
)

axes[1].axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)

axes[1].set_ylabel(
    r"$U_{\mathrm{along}}$"
    "\n(m/s)"
)

axes[1].grid(
    True,
    alpha=0.3,
)


# Speed
axes[2].plot(
    df["time"],
    df["depth_avg_current_speed_m_s"],
)

axes[2].set_ylim(
    bottom=0
)

axes[2].set_ylabel(
    r"$|\mathbf{U}_{DA}|$"
    "\n(m/s)"
)

axes[2].set_xlabel(
    "Time"
)

axes[2].grid(
    True,
    alpha=0.3,
)


fig.suptitle(
    "Depth-averaged background current\n"
    "Cross-shore / alongshore coordinates"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.show()