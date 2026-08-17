# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:57:46 2026

@author: WangX3
Inspect ADCP velocity data and assess whether each vertical cell
is suitable for burst-based spectral analysis.
"""

import h5py
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADCP"
    r"\UP\adcp_dvn_201804_f1p1_000.nc"
)

velocity_variables = [
    "velocity_east",
    "velocity_north",
    "velocity_up",
]

height_variable = "z_measures"

expected_sampling_frequency_hz = 1.25

# Use relatively small reads so damaged HDF5 regions can be
# localized rather than losing a huge section at once.
read_block_samples = 10_000

diagnostic_window_minutes = 30.0

# Velocity values larger than this magnitude are considered
# clearly suspicious for this application.
maximum_abs_velocity_m_s = 10.0

# A 30-min velocity record with smaller variability than this
# is flagged as nearly frozen.
minimum_window_std_m_s = 1e-4

minimum_window_valid_fraction = 0.98

minimum_overall_valid_fraction = 0.90

number_of_gaps_to_report = 20


# ============================================================
# PRINT VARIABLES AND ATTRIBUTES
# ============================================================

with h5py.File(file_path, "r") as f:
    variables = list(f.keys())


print("Variables found:")

for name in variables:
    print(" ", name)


with h5py.File(file_path, "r") as f:

    for variable_name in variables:

        print("\n" + "=" * 70)
        print(variable_name)

        try:
            dset = f[variable_name]

            print("shape:", dset.shape)
            print("dtype:", dset.dtype)
            print("chunks:", dset.chunks)
            print("compression:", dset.compression)

        except Exception as exc:

            print(
                "Could not inspect dataset:",
                type(exc).__name__,
                exc,
            )

            continue

        print("Attributes:")

        try:
            attributes = list(
                dset.attrs.keys()
            )

        except Exception as exc:

            print(
                "Could not list attributes:",
                type(exc).__name__,
                exc,
            )

            attributes = []

        for attr_name in attributes:

            try:
                print(
                    " ",
                    attr_name,
                    "=",
                    dset.attrs[attr_name],
                )

            except Exception as exc:

                print(
                    " ",
                    attr_name,
                    "FAILED:",
                    type(exc).__name__,
                    exc,
                )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def scalar_attribute(
    dataset,
    name,
    default,
):
    """Return scalar HDF5 attribute as float."""

    value = dataset.attrs.get(
        name,
        default,
    )

    array = np.asarray(
        value
    )

    if array.size != 1:
        raise ValueError(
            f"Attribute '{name}' is not scalar."
        )

    return float(
        array.squeeze()
    )


def get_fill_values(dataset):
    """Return explicit _FillValue / missing_value values."""

    values = []

    for name in [
        "_FillValue",
        "missing_value",
    ]:

        if name not in dataset.attrs:
            continue

        attribute = np.asarray(
            dataset.attrs[name]
        ).ravel()

        for value in attribute:

            try:
                values.append(
                    float(value)
                )
            except Exception:
                pass

    return values


def decode_velocity_block(
    raw,
    dataset,
):
    """
    Convert packed velocity values to m/s and flag likely
    invalid values.
    """

    raw = np.asarray(
        raw
    )

    raw_float = raw.astype(
        np.float64
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

    # --------------------------------------------------------
    # Explicit fill values
    # --------------------------------------------------------

    explicit_fill = np.zeros(
        raw.shape,
        dtype=bool,
    )

    for fill_value in get_fill_values(
        dataset
    ):

        explicit_fill |= (
            raw_float == fill_value
        )

    # --------------------------------------------------------
    # Common integer sentinel values
    # --------------------------------------------------------

    integer_sentinel = np.zeros(
        raw.shape,
        dtype=bool,
    )

    if np.issubdtype(
        dataset.dtype,
        np.signedinteger,
    ):

        info = np.iinfo(
            dataset.dtype
        )

        integer_sentinel |= (
            raw_float <= info.min + 1
        )

    elif np.issubdtype(
        dataset.dtype,
        np.unsignedinteger,
    ):

        info = np.iinfo(
            dataset.dtype
        )

        integer_sentinel |= (
            raw_float >= info.max - 1
        )

    # --------------------------------------------------------
    # Convert to physical velocity
    # --------------------------------------------------------

    velocity = (
        raw_float * scale
        + offset
    )

    huge_mask = (
        np.abs(velocity)
        > maximum_abs_velocity_m_s
    )

    nonfinite_mask = (
        ~np.isfinite(velocity)
    )

    invalid = (
        explicit_fill
        | integer_sentinel
        | huge_mask
        | nonfinite_mask
    )

    velocity[
        invalid
    ] = np.nan

    counts = {
        "explicit_fill":
            int(
                explicit_fill.sum()
            ),
        "integer_sentinel":
            int(
                integer_sentinel.sum()
            ),
        "huge":
            int(
                huge_mask.sum()
            ),
        "nonfinite":
            int(
                nonfinite_mask.sum()
            ),
    }

    return velocity, counts


def contiguous_true_runs(mask):
    """Return contiguous True runs."""

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.size == 0:

        return pd.DataFrame(
            columns=[
                "start_index",
                "stop_index",
                "sample_count",
            ]
        )

    padded = np.concatenate(
        [
            [False],
            mask,
            [False],
        ]
    )

    changes = np.diff(
        padded.astype(np.int8)
    )

    starts = np.where(
        changes == 1
    )[0]

    stops = np.where(
        changes == -1
    )[0]

    return pd.DataFrame(
        {
            "start_index":
                starts,
            "stop_index":
                stops,
            "sample_count":
                stops - starts,
        }
    )


# ============================================================
# BASIC STRUCTURE
# ============================================================

with h5py.File(file_path, "r") as f:

    if "time" not in f:
        raise KeyError(
            "No 'time' variable found."
        )

    if height_variable not in f:
        raise KeyError(
            f"No '{height_variable}' variable found."
        )

    for variable_name in velocity_variables:

        if variable_name not in f:
            raise KeyError(
                f"No '{variable_name}' variable found."
            )

    time_ds = f["time"]

    number_of_samples = (
        time_ds.shape[0]
    )

    z_measures = np.asarray(
        f[height_variable][...],
        dtype=float,
    ).ravel()

    number_of_cells = len(
        z_measures
    )

    for variable_name in velocity_variables:

        ds = f[
            variable_name
        ]

        if ds.shape != (
            number_of_samples,
            number_of_cells,
        ):

            raise ValueError(
                f"{variable_name} has shape "
                f"{ds.shape}, expected "
                f"({number_of_samples}, "
                f"{number_of_cells})."
            )


print("\n" + "=" * 70)
print("FILE STRUCTURE")
print("=" * 70)

print(
    "Number of time samples:",
    number_of_samples,
)

print(
    "Number of vertical cells:",
    number_of_cells,
)

print(
    "z_measures (m):"
)

print(
    z_measures
)


# ============================================================
# TIME DIAGNOSTICS
# ============================================================

with h5py.File(file_path, "r") as f:

    time_ds = f["time"]

    time_scale = scalar_attribute(
        time_ds,
        "scale_factor",
        1.0,
    )

    time_offset = scalar_attribute(
        time_ds,
        "add_offset",
        0.0,
    )

    first_seconds = (
        float(
            time_ds[0]
        )
        * time_scale
        + time_offset
    )

    last_seconds = (
        float(
            time_ds[-1]
        )
        * time_scale
        + time_offset
    )

    first_time = pd.to_datetime(
        first_seconds,
        unit="s",
        utc=True,
    )

    last_time = pd.to_datetime(
        last_seconds,
        unit="s",
        utc=True,
    )

    wall_clock_duration_days = (
        last_seconds
        - first_seconds
    ) / 86400.0

    positive_dt_parts = []

    nonpositive_dt_count = 0

    previous_time = None

    for start in range(
        0,
        number_of_samples,
        read_block_samples,
    ):

        stop = min(
            start + read_block_samples,
            number_of_samples,
        )

        try:

            raw_time = (
                time_ds[
                    start:stop
                ]
                .astype(np.float64)
            )

        except Exception as exc:

            print(
                "TIME READ FAILED:",
                start,
                stop,
                type(exc).__name__,
                exc,
            )

            previous_time = None

            continue

        seconds = (
            raw_time * time_scale
            + time_offset
        )

        if previous_time is not None:

            seconds_for_diff = np.concatenate(
                [
                    [previous_time],
                    seconds,
                ]
            )

        else:

            seconds_for_diff = seconds

        dt = np.diff(
            seconds_for_diff
        )

        positive_dt_parts.append(
            dt[
                np.isfinite(dt)
                & (dt > 0)
            ]
        )

        nonpositive_dt_count += int(
            np.sum(
                np.isfinite(dt)
                & (dt <= 0)
            )
        )

        previous_time = (
            seconds[-1]
        )


if not positive_dt_parts:

    raise RuntimeError(
        "Could not obtain usable timestamp intervals."
    )


positive_dt = np.concatenate(
    positive_dt_parts
)

q25_dt = np.nanpercentile(
    positive_dt,
    25,
)

regular_dt = positive_dt[
    positive_dt
    <= q25_dt * 1.5
]

if regular_dt.size == 0:

    regular_dt = positive_dt

median_dt = float(
    np.nanmedian(
        regular_dt
    )
)

detected_fs = (
    1.0 / median_dt
)

gap_threshold_seconds = (
    median_dt * 1.5
)

time_gap_values = positive_dt[
    positive_dt
    > gap_threshold_seconds
]


print("\n" + "=" * 70)
print("TIME DIAGNOSTICS")
print("=" * 70)

print(
    "First timestamp:",
    first_time,
)

print(
    "Last timestamp:",
    last_time,
)

print(
    f"Wall-clock duration: "
    f"{wall_clock_duration_days:.3f} days"
)

print(
    "Median regular dt:",
    median_dt,
    "s",
)

print(
    "Detected sampling frequency:",
    detected_fs,
    "Hz",
)

print(
    "Expected sampling frequency:",
    expected_sampling_frequency_hz,
    "Hz",
)

print(
    "Nonpositive timestamp intervals:",
    nonpositive_dt_count,
)

print(
    "Number of timestamp gaps:",
    len(time_gap_values),
)

if len(time_gap_values) > 0:

    print(
        "Largest timestamp gap:",
        np.nanmax(
            time_gap_values
        ),
        "s",
    )


# ============================================================
# READ VELOCITY COMPONENTS
# ============================================================

velocity_data = {}

read_failure_records = []

invalid_counts = {}


for variable_name in velocity_variables:

    print("\n" + "=" * 70)
    print(
        "READING",
        variable_name,
    )
    print("=" * 70)

    values_all = np.full(
        (
            number_of_samples,
            number_of_cells,
        ),
        np.nan,
        dtype=np.float64,
    )

    invalid_counts[
        variable_name
    ] = {
        "explicit_fill": 0,
        "integer_sentinel": 0,
        "huge": 0,
        "nonfinite": 0,
    }

    with h5py.File(file_path, "r") as f:

        ds = f[
            variable_name
        ]

        for start in range(
            0,
            number_of_samples,
            read_block_samples,
        ):

            stop = min(
                start + read_block_samples,
                number_of_samples,
            )

            try:

                raw = ds[
                    start:stop,
                    :
                ]

                values, counts = (
                    decode_velocity_block(
                        raw,
                        ds,
                    )
                )

                values_all[
                    start:stop,
                    :
                ] = values

                for key in invalid_counts[
                    variable_name
                ]:

                    invalid_counts[
                        variable_name
                    ][key] += counts[
                        key
                    ]

            except Exception as exc:

                read_failure_records.append(
                    {
                        "variable":
                            variable_name,
                        "start_index":
                            start,
                        "stop_index":
                            stop,
                        "sample_count":
                            stop - start,
                        "error":
                            (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                    }
                )

    velocity_data[
        variable_name
    ] = values_all


# ============================================================
# HDF5 READ STATUS
# ============================================================

print("\n" + "=" * 70)
print("HDF5 READ STATUS")
print("=" * 70)


if not read_failure_records:

    print(
        "No HDF5 velocity read failures detected."
    )

else:

    failures = pd.DataFrame(
        read_failure_records
    )

    print(
        "Number of failed read blocks:",
        len(failures),
    )

    print(
        failures.head(
            number_of_gaps_to_report
        ).to_string(
            index=False
        )
    )


# ============================================================
# INVALID-VALUE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("INVALID-VALUE MECHANISMS")
print("=" * 70)


for variable_name in velocity_variables:

    print(
        "\n",
        variable_name,
    )

    for key, count in invalid_counts[
        variable_name
    ].items():

        print(
            f"  {key:18s}: "
            f"{count}"
        )


# ============================================================
# PER-CELL STATISTICS
# ============================================================

cell_records = []


for cell_index in range(
    number_of_cells
):

    cell_number = (
        cell_index + 1
    )

    z_m = (
        z_measures[
            cell_index
        ]
    )

    for variable_name in velocity_variables:

        values = velocity_data[
            variable_name
        ][
            :,
            cell_index
        ]

        finite = np.isfinite(
            values
        )

        valid_count = int(
            finite.sum()
        )

        valid_fraction = (
            valid_count
            / number_of_samples
        )

        zero_fraction = float(
            np.mean(
                values[
                    finite
                ] == 0
            )
        ) if valid_count else np.nan

        if valid_count > 0:

            valid_values = values[
                finite
            ]

            (
                minimum,
                q25,
                median,
                q75,
                maximum,
            ) = np.percentile(
                valid_values,
                [
                    0,
                    25,
                    50,
                    75,
                    100,
                ],
            )

            mean_value = float(
                np.mean(
                    valid_values
                )
            )

            std_value = float(
                np.std(
                    valid_values
                )
            )

        else:

            minimum = np.nan
            q25 = np.nan
            median = np.nan
            q75 = np.nan
            maximum = np.nan
            mean_value = np.nan
            std_value = np.nan

        cell_records.append(
            {
                "cell_number":
                    cell_number,
                "z_m":
                    z_m,
                "variable":
                    variable_name,
                "valid_fraction":
                    valid_fraction,
                "zero_fraction":
                    zero_fraction,
                "minimum_m_s":
                    minimum,
                "q25_m_s":
                    q25,
                "median_m_s":
                    median,
                "q75_m_s":
                    q75,
                "maximum_m_s":
                    maximum,
                "mean_m_s":
                    mean_value,
                "std_m_s":
                    std_value,
            }
        )


cell_statistics = pd.DataFrame(
    cell_records
)


print("\n" + "=" * 70)
print("PER-CELL VELOCITY STATISTICS")
print("=" * 70)

print(
    cell_statistics.to_string(
        index=False
    )
)


# ============================================================
# 30-MIN WINDOW QC
# ============================================================

window_samples = int(
    round(
        diagnostic_window_minutes
        * 60.0
        * detected_fs
    )
)


window_records = []


for cell_index in range(
    number_of_cells
):

    cell_number = (
        cell_index + 1
    )

    z_m = (
        z_measures[
            cell_index
        ]
    )

    east = velocity_data[
        "velocity_east"
    ][
        :,
        cell_index
    ]

    north = velocity_data[
        "velocity_north"
    ][
        :,
        cell_index
    ]

    up = velocity_data[
        "velocity_up"
    ][
        :,
        cell_index
    ]

    for start in range(
        0,
        number_of_samples,
        window_samples,
    ):

        stop = min(
            start + window_samples,
            number_of_samples,
        )

        east_window = east[
            start:stop
        ]

        north_window = north[
            start:stop
        ]

        up_window = up[
            start:stop
        ]

        # Require all three components to be finite
        # at the same timestamp.
        finite = (
            np.isfinite(
                east_window
            )
            & np.isfinite(
                north_window
            )
            & np.isfinite(
                up_window
            )
        )

        valid_fraction_window = float(
            finite.mean()
        )

        if finite.any():

            east_std = float(
                np.std(
                    east_window[
                        finite
                    ]
                )
            )

            north_std = float(
                np.std(
                    north_window[
                        finite
                    ]
                )
            )

            up_std = float(
                np.std(
                    up_window[
                        finite
                    ]
                )
            )

        else:

            east_std = np.nan
            north_std = np.nan
            up_std = np.nan

        all_invalid = (
            finite.sum() == 0
        )

        nearly_frozen = (
            finite.any()
            and (
                east_std
                < minimum_window_std_m_s
                or north_std
                < minimum_window_std_m_s
                or up_std
                < minimum_window_std_m_s
            )
        )

        usable_for_spectrum = (
            valid_fraction_window
            >= minimum_window_valid_fraction
            and not nearly_frozen
        )

        window_records.append(
            {
                "cell_number":
                    cell_number,
                "z_m":
                    z_m,
                "start_index":
                    start,
                "stop_index":
                    stop,
                "valid_fraction":
                    valid_fraction_window,
                "east_std_m_s":
                    east_std,
                "north_std_m_s":
                    north_std,
                "up_std_m_s":
                    up_std,
                "all_invalid":
                    all_invalid,
                "nearly_frozen":
                    nearly_frozen,
                "usable_for_spectrum":
                    usable_for_spectrum,
            }
        )


windows = pd.DataFrame(
    window_records
)


# ============================================================
# PER-CELL WINDOW SUMMARY
# ============================================================

window_summary = (
    windows
    .groupby(
        [
            "cell_number",
            "z_m",
        ],
        as_index=False,
    )
    .agg(
        total_windows=(
            "usable_for_spectrum",
            "size",
        ),
        all_invalid_windows=(
            "all_invalid",
            "sum",
        ),
        nearly_frozen_windows=(
            "nearly_frozen",
            "sum",
        ),
        usable_windows=(
            "usable_for_spectrum",
            "sum",
        ),
        usable_fraction=(
            "usable_for_spectrum",
            "mean",
        ),
    )
)


print("\n" + "=" * 70)
print("30-MINUTE WINDOW QC BY CELL")
print("=" * 70)

print(
    window_summary.to_string(
        index=False
    )
)


# ============================================================
# CONTIGUOUS INVALID RUNS BY CELL
# ============================================================

invalid_run_records = []


for cell_index in range(
    number_of_cells
):

    cell_number = (
        cell_index + 1
    )

    z_m = (
        z_measures[
            cell_index
        ]
    )

    east = velocity_data[
        "velocity_east"
    ][
        :,
        cell_index
    ]

    north = velocity_data[
        "velocity_north"
    ][
        :,
        cell_index
    ]

    up = velocity_data[
        "velocity_up"
    ][
        :,
        cell_index
    ]

    invalid = ~(
        np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(up)
    )

    runs = contiguous_true_runs(
        invalid
    )

    if runs.empty:
        continue

    runs[
        "duration_seconds"
    ] = (
        runs[
            "sample_count"
        ]
        / detected_fs
    )

    runs[
        "duration_minutes"
    ] = (
        runs[
            "duration_seconds"
        ]
        / 60.0
    )

    runs[
        "duration_hours"
    ] = (
        runs[
            "duration_seconds"
        ]
        / 3600.0
    )

    runs[
        "cell_number"
    ] = (
        cell_number
    )

    runs[
        "z_m"
    ] = (
        z_m
    )

    invalid_run_records.append(
        runs
    )


if invalid_run_records:

    invalid_runs = pd.concat(
        invalid_run_records,
        ignore_index=True,
    )

    invalid_runs = (
        invalid_runs
        .sort_values(
            "sample_count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

else:

    invalid_runs = pd.DataFrame()


print("\n" + "=" * 70)
print("LONGEST INVALID VELOCITY RUNS")
print("=" * 70)


if invalid_runs.empty:

    print(
        "No invalid velocity runs detected."
    )

else:

    print(
        invalid_runs.head(
            number_of_gaps_to_report
        ).to_string(
            index=False
        )
    )


# ============================================================
# FINAL CELL ASSESSMENT
# ============================================================

assessment = window_summary.copy()


def classify_cell(row):

    if row[
        "usable_windows"
    ] == 0:

        return (
            "NOT USABLE"
        )

    if row[
        "usable_fraction"
    ] < 0.50:

        return (
            "POOR / LIMITED"
        )

    if row[
        "usable_fraction"
    ] < 0.80:

        return (
            "PARTIALLY USABLE"
        )

    return (
        "GENERALLY USABLE"
    )


assessment[
    "assessment"
] = assessment.apply(
    classify_cell,
    axis=1,
)


print("\n" + "=" * 70)
print("VELOCITY SPECTRUM ASSESSMENT BY CELL")
print("=" * 70)

print(
    assessment.to_string(
        index=False
    )
)