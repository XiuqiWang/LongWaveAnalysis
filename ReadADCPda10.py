# -*- coding: utf-8 -*-
"""
Read processed ADCP 10-min depth-averaged mean velocity.

Workflow
--------
1. Read:
       velocity_dabmean
       water_depth
       time

2. Decode scale factors.

3. Identify no-measurement records:
       - non-finite water depth
       - water depth below a physically reasonable deployment threshold

4. Convert no-measurement records to NaN.

5. Interpolate ONLY short internal gaps (e.g. <= 30 min).
   Long gaps and the beginning/end outside the deployment are NOT filled.

6. Plot original East/North velocity and water depth.

7. Rotate CLEANED/interpolated East/North velocity to
   cross-shore/alongshore coordinates.

8. Save:
       time
       cross-shore velocity
       alongshore velocity
       current speed
       water depth
   to CSV.
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
    r"\UP\adcp_dvn_201804_f3p3_000_da10.nc"
)

case_id = "DVN_F3_ADCP_upward"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    output_folder
    / f"{case_id}_depth_averaged_current.csv"
)

# ------------------------------------------------------------
# Deployment/QC settings
# ------------------------------------------------------------

# Values around 0 or ~2.3 m correspond to no valid deployment data.
minimum_valid_water_depth_m = 10.0

# Data are nominally 10-min averaged.
# Interpolate only gaps up to 30 min.
maximum_interpolation_gap_samples = 3


# ------------------------------------------------------------
# Cross-shore coordinate definition
# ------------------------------------------------------------

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
    """Read scalar HDF5 attribute."""

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
    Read complete HDF5 variable and apply
    scale_factor/add_offset.
    """

    raw = np.asarray(
        dataset[...],
        dtype=np.float64,
    )

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
                raw
                == float(fill_value)
            )

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


def interpolate_short_internal_gaps(
    dataframe,
    columns,
    maximum_gap_samples,
):
    """
    Interpolate only NaN runs whose length is <= maximum_gap_samples.

    Important:
    - gaps at beginning/end are NOT interpolated
    - long internal gaps are NOT partially interpolated
    """

    result = dataframe.copy()

    result = (
        result
        .set_index("time")
    )

    for column in columns:

        series = result[column].copy()

        missing = series.isna()

        if not missing.any():
            continue

        # Candidate interpolation over all internal gaps.
        interpolated = (
            series
            .interpolate(
                method="time",
                limit_area="inside",
            )
        )

        # Identify consecutive missing runs.
        group_number = (
            missing
            .ne(missing.shift())
            .cumsum()
        )

        groups = (
            pd.DataFrame(
                {
                    "missing": missing,
                    "group": group_number,
                },
                index=series.index,
            )
            .loc[missing]
            .groupby("group")
        )

        for _, group in groups:

            gap_index = group.index

            gap_length = len(
                gap_index
            )

            if (
                gap_length
                > maximum_gap_samples
            ):
                continue

            first_position = (
                series.index
                .get_loc(
                    gap_index[0]
                )
            )

            last_position = (
                series.index
                .get_loc(
                    gap_index[-1]
                )
            )

            # Do not extrapolate at beginning/end.
            if (
                first_position == 0
                or last_position
                == len(series) - 1
            ):
                continue

            before = series.iloc[
                first_position - 1
            ]

            after = series.iloc[
                last_position + 1
            ]

            if (
                np.isfinite(before)
                and np.isfinite(after)
            ):

                series.loc[
                    gap_index
                ] = interpolated.loc[
                    gap_index
                ]

        result[column] = series

    return (
        result
        .reset_index()
    )


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

    lat1, lon1 = map(
        float,
        point_from,
    )

    lat2, lon2 = map(
        float,
        point_to,
    )

    earth_radius_m = (
        6_371_000.0
    )

    lat1_rad = np.deg2rad(
        lat1
    )

    lat2_rad = np.deg2rad(
        lat2
    )

    lon1_rad = np.deg2rad(
        lon1
    )

    lon2_rad = np.deg2rad(
        lon2
    )

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
        * np.cos(
            mean_lat_rad
        )
    )

    delta_north_m = (
        earth_radius_m
        * (
            lat2_rad
            - lat1_rad
        )
    )

    transect_length_m = (
        np.hypot(
            delta_east_m,
            delta_north_m,
        )
    )

    if transect_length_m == 0:

        raise ValueError(
            "The two transect points are identical."
        )

    # Positive cross-shore unit vector.
    cross_east = (
        delta_east_m
        / transect_length_m
    )

    cross_north = (
        delta_north_m
        / transect_length_m
    )

    # Positive alongshore:
    # 90° counterclockwise.
    along_east = (
        -cross_north
    )

    along_north = (
        cross_east
    )

    cross_shore = (
        east * cross_east
        + north * cross_north
    )

    alongshore = (
        east * along_east
        + north * along_north
    )

    bearing_deg = (
        np.degrees(
            np.arctan2(
                delta_east_m,
                delta_north_m,
            )
        )
        + 360
    ) % 360

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

    time_raw = (
        read_scaled_variable(
            f["time"]
        )
    )

    time = pd.to_datetime(
        time_raw,
        unit="s",
        utc=True,
        errors="coerce",
    )

    velocity_dabmean = (
        read_scaled_variable(
            f[
                "velocity_dabmean"
            ]
        )
    )

    if (
        velocity_dabmean.shape[0]
        != 3
    ):

        raise ValueError(
            "Expected velocity_dabmean "
            "shape (3,time), got "
            f"{velocity_dabmean.shape}"
        )

    velocity_east = (
        velocity_dabmean[
            0,
            :
        ]
    )

    velocity_north = (
        velocity_dabmean[
            1,
            :
        ]
    )

    velocity_up = (
        velocity_dabmean[
            2,
            :
        ]
    )

    water_depth = (
        read_scaled_variable(
            f["water_depth"]
        )
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


# ============================================================
# TIME INFORMATION
# ============================================================

dt_seconds = (
    df["time"]
    .diff()
    .dt.total_seconds()
    .median()
)

print(
    "\nStart:",
    df["time"].iloc[0],
)

print(
    "End:",
    df["time"].iloc[-1],
)

print(
    "Rows:",
    len(df),
)

print(
    "Median sampling interval:",
    dt_seconds / 60,
    "min",
)


# ============================================================
# IDENTIFY VALID ADCP DEPLOYMENT RECORDS
# ============================================================

# Do NOT use East/North == 0 to identify missing data:
# a real tidal current can cross zero.
#
# Instead use water depth as the measurement-state indicator.

valid_measurement = (
    np.isfinite(
        df["water_depth_m"]
    )
    & (
        df["water_depth_m"]
        >= minimum_valid_water_depth_m
    )
)

print(
    "\nValid measured records:",
    valid_measurement.sum(),
)

print(
    "Invalid/no-measurement records:",
    (~valid_measurement).sum(),
)


# ============================================================
# DETERMINE DEPLOYMENT PERIOD
# ============================================================

if not valid_measurement.any():

    raise RuntimeError(
        "No valid ADCP deployment "
        "measurements found."
    )

first_valid_index = (
    np.flatnonzero(
        valid_measurement
    )[0]
)

last_valid_index = (
    np.flatnonzero(
        valid_measurement
    )[-1]
)

deployment_start = (
    df.loc[
        first_valid_index,
        "time",
    ]
)

deployment_end = (
    df.loc[
        last_valid_index,
        "time",
    ]
)

print(
    "\nDetected deployment start:",
    deployment_start,
)

print(
    "Detected deployment end:",
    deployment_end,
)


# ============================================================
# REMOVE PRE-/POST-DEPLOYMENT DATA
# ============================================================

# This removes the ~2.3-m water-depth values at the
# beginning and end instead of trying to interpolate them.

df = (
    df.loc[
        first_valid_index:
        last_valid_index
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# Recalculate validity for trimmed record.
valid_measurement = (
    np.isfinite(
        df["water_depth_m"]
    )
    & (
        df["water_depth_m"]
        >= minimum_valid_water_depth_m
    )
)


# ============================================================
# MASK NO-MEASUREMENT PERIODS
# ============================================================

columns_to_mask = [
    "depth_avg_east_m_s",
    "depth_avg_north_m_s",
    "depth_avg_up_m_s",
    "water_depth_m",
]

df.loc[
    ~valid_measurement,
    columns_to_mask,
] = np.nan


# ============================================================
# REPORT GAP LENGTHS
# ============================================================

missing = (
    ~valid_measurement
)

missing_groups = (
    missing
    .ne(
        missing.shift()
    )
    .cumsum()
)

gap_lengths = (
    df.loc[
        missing
    ]
    .groupby(
        missing_groups[
            missing
        ]
    )
    .size()
)

print(
    "\nNo-measurement gap lengths:"
)

if len(gap_lengths) > 0:

    for n, count in (
        gap_lengths
        .value_counts()
        .sort_index()
        .items()
    ):

        print(
            f"{n} samples = "
            f"{n * dt_seconds / 60:.1f} min "
            f"({count} occurrences)"
        )

else:

    print(
        "No gaps detected."
    )


# ============================================================
# INTERPOLATE ONLY SHORT INTERNAL GAPS
# ============================================================

interpolation_columns = [
    "depth_avg_east_m_s",
    "depth_avg_north_m_s",
    "depth_avg_up_m_s",
    "water_depth_m",
]

df_processed = (
    interpolate_short_internal_gaps(
        dataframe=df,
        columns=interpolation_columns,
        maximum_gap_samples=
            maximum_interpolation_gap_samples,
    )
)


# ============================================================
# FIGURE 1
# ORIGINAL EAST/NORTH + WATER DEPTH
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(
        13,
        7,
    ),
    sharex=True,
)

axes[0].plot(
    df_processed["time"],
    df_processed[
        "depth_avg_east_m_s"
    ],
    linewidth=0.8,
    label=r"$U_E$",
)

axes[0].plot(
    df_processed["time"],
    df_processed[
        "depth_avg_north_m_s"
    ],
    linewidth=0.8,
    label=r"$U_N$",
)

axes[0].axhline(
    0,
    linewidth=0.6,
    linestyle="--",
)

axes[0].set_ylabel(
    "Depth-averaged\nvelocity (m/s)"
)

axes[0].legend()

axes[0].grid(
    True,
    alpha=0.25,
)


axes[1].plot(
    df_processed["time"],
    df_processed[
        "water_depth_m"
    ],
    linewidth=0.8,
)

axes[1].set_ylabel(
    "Water depth (m)"
)

axes[1].set_xlabel(
    "Time"
)

axes[1].grid(
    True,
    alpha=0.25,
)


fig.suptitle(
    "ADCP depth-averaged velocity and water depth\n"
    f"{case_id}"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

plt.show()


# ============================================================
# ROTATE THE CLEANED / INTERPOLATED VELOCITIES
# ============================================================

# IMPORTANT:
# Rotate df_processed, NOT the original raw df.

(
    df_processed[
        "depth_avg_cross_shore_m_s"
    ],
    df_processed[
        "depth_avg_alongshore_m_s"
    ],
    rotation_info,
) = rotate_enu_to_cross_along(
    east=df_processed[
        "depth_avg_east_m_s"
    ],
    north=df_processed[
        "depth_avg_north_m_s"
    ],
    point_from=
        nearshore_point,
    point_to=
        offshore_point,
)


# ============================================================
# CURRENT SPEED
# ============================================================

df_processed[
    "depth_avg_current_speed_m_s"
] = np.hypot(
    df_processed[
        "depth_avg_cross_shore_m_s"
    ],
    df_processed[
        "depth_avg_alongshore_m_s"
    ],
)


# ============================================================
# ROTATION INFORMATION
# ============================================================

print(
    "\nRotation information:"
)

for key, value in (
    rotation_info.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# VERIFY MAGNITUDE IS PRESERVED
# ============================================================

speed_enu = np.hypot(
    df_processed[
        "depth_avg_east_m_s"
    ],
    df_processed[
        "depth_avg_north_m_s"
    ],
)

speed_rotated = np.hypot(
    df_processed[
        "depth_avg_cross_shore_m_s"
    ],
    df_processed[
        "depth_avg_alongshore_m_s"
    ],
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
# ROTATED DEPTH-AVERAGED CURRENT
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(
        13,
        9,
    ),
    sharex=True,
)


# Cross-shore
axes[0].plot(
    df_processed["time"],
    df_processed[
        "depth_avg_cross_shore_m_s"
    ],
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
    df_processed["time"],
    df_processed[
        "depth_avg_alongshore_m_s"
    ],
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


# Current speed
axes[2].plot(
    df_processed["time"],
    df_processed[
        "depth_avg_current_speed_m_s"
    ],
)

axes[2].set_ylim(
    bottom=0
)

axes[2].set_ylabel(
    r"$|\mathbf{U}_{DA}|$"
    "\n(m/s)"
)

axes[2].grid(
    True,
    alpha=0.3,
)


# Water depth
axes[3].plot(
    df_processed["time"],
    df_processed[
        "water_depth_m"
    ],
)

axes[3].set_ylabel(
    r"$h$ (m)"
)

axes[3].set_xlabel(
    "Time"
)

axes[3].grid(
    True,
    alpha=0.3,
)


fig.suptitle(
    "Depth-averaged background current\n"
    f"{case_id}"
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
# SAVE CSV
# ============================================================

columns_to_save = [
    "time",
    "depth_avg_cross_shore_m_s",
    "depth_avg_alongshore_m_s",
    "depth_avg_current_speed_m_s",
    "water_depth_m",
]

output = (
    df_processed[
        columns_to_save
    ]
    .copy()
)


# Do not save rows for which the long-gap interpolation
# deliberately left all current information missing.
output = (
    output
    .dropna(
        subset=[
            "depth_avg_cross_shore_m_s",
            "depth_avg_alongshore_m_s",
            "water_depth_m",
        ],
        how="all",
    )
    .reset_index(
        drop=True
    )
)


output.to_csv(
    output_file,
    index=False,
)


print(
    "\nSaved processed "
    "depth-averaged current:"
)

print(
    output_file
)

print(
    "Rows saved:",
    len(output),
)

print(
    "Start:",
    output["time"].iloc[0],
)

print(
    "End:",
    output["time"].iloc[-1],
)

print(
    "\nMissing values in saved file:"
)

print(
    output.isna().sum()
)