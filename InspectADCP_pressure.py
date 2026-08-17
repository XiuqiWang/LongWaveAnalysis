# -*- coding: utf-8 -*-

"""
Inspect ADCP atmospheric-pressure-corrected pressure
(pressure_corrected) and assess whether it is suitable for
surface-elevation spectral analysis.
"""

import h5py
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADCP"
    r"\UP\adcp_dvn_201804_f4p2_000.nc"
)

pressure_variable = "pressure_corrected"

# Change if necessary according to deployment documentation.
expected_sampling_frequency_hz = 1.25

# Number of samples attempted in each HDF5 read.
#
# This file has previously shown possible HDF5/B-tree read
# problems, so keep this reasonably small.
read_block_samples = 10_000

# Window duration corresponding to your spectral analysis.
diagnostic_window_minutes = 30.0

# Atmospheric-corrected pressure at a submerged sensor should
# normally be > 0 Pa.
treat_zero_as_invalid = True

# Very large pressure values are regarded as sentinel-like /
# nonphysical.
maximum_abs_pressure_pa = 1.0e7

# Flag essentially constant 30-min signals.
minimum_window_std_pa = 1.0

# Minimum finite-data coverage for spectral analysis.
minimum_window_valid_fraction = 0.98

# Only used for the final overall assessment.
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
# HELPERS
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
    """
    Return explicit _FillValue and missing_value values.
    """

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


def decode_pressure_block(
    raw,
    dataset,
):
    """
    Convert packed pressure_corrected values to physical Pa.

    Also identify several possible invalid-value conventions.
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
    # Explicit fill / missing values
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
    # Possible integer sentinel values
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

        # For int32:
        # catches -2147483648 and -2147483647.
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
    # Convert packed values
    # --------------------------------------------------------

    pressure = (
        raw_float * scale
        + offset
    )


    zero_mask = (
        pressure == 0
    )

    huge_mask = (
        np.abs(pressure)
        > maximum_abs_pressure_pa
    )

    nonfinite_mask = (
        ~np.isfinite(pressure)
    )


    invalid = (
        explicit_fill
        | integer_sentinel
        | huge_mask
        | nonfinite_mask
    )


    if treat_zero_as_invalid:

        invalid |= (
            zero_mask
        )


    pressure[
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
        "zero":
            int(
                zero_mask.sum()
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


    return pressure, counts


def contiguous_true_runs(mask):
    """
    Find contiguous True regions.
    """

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
# BASIC FILE STRUCTURE
# ============================================================

with h5py.File(file_path, "r") as f:

    if "time" not in f:

        raise KeyError(
            "No 'time' variable found."
        )

    if pressure_variable not in f:

        raise KeyError(
            f"No '{pressure_variable}' variable found."
        )


    time_ds = f["time"]

    pressure_ds = f[
        pressure_variable
    ]


    number_of_samples = (
        time_ds.shape[0]
    )


    if pressure_ds.ndim != 1:

        raise ValueError(
            "Expected pressure_corrected to have "
            f"shape (time,), found {pressure_ds.shape}."
        )


    if pressure_ds.shape[0] != number_of_samples:

        raise ValueError(
            "Pressure and time lengths differ."
        )


    pressure_dtype = (
        pressure_ds.dtype
    )


    pressure_scale = (
        scalar_attribute(
            pressure_ds,
            "scale_factor",
            1.0,
        )
    )


    pressure_offset = (
        scalar_attribute(
            pressure_ds,
            "add_offset",
            0.0,
        )
    )


    pressure_fill_values = (
        get_fill_values(
            pressure_ds
        )
    )


    # --------------------------------------------------------
    # Pressure sensor vertical metadata
    # --------------------------------------------------------

    if (
        "sensor_depth"
        in pressure_ds.attrs
    ):

        pressure_sensor_depth_m = (
            scalar_attribute(
                pressure_ds,
                "sensor_depth",
                np.nan,
            )
        )

    else:

        pressure_sensor_depth_m = (
            np.nan
        )


    if (
        "depth_offset"
        in pressure_ds.attrs
    ):

        depth_offset_m = (
            scalar_attribute(
                pressure_ds,
                "depth_offset",
                np.nan,
            )
        )

    else:

        depth_offset_m = (
            np.nan
        )


print("\n" + "=" * 70)
print("FILE STRUCTURE")
print("=" * 70)

print(
    "Number of samples:",
    number_of_samples,
)

print(
    "Pressure shape:",
    (number_of_samples,),
)

print(
    "Pressure dtype:",
    pressure_dtype,
)

print(
    "Scale factor:",
    pressure_scale,
)

print(
    "Add offset:",
    pressure_offset,
)

print(
    "Explicit fill values:",
    pressure_fill_values,
)

print(
    "Pressure sensor_depth attribute:",
    pressure_sensor_depth_m,
    "m",
)

print(
    "Pressure depth_offset attribute:",
    depth_offset_m,
    "m",
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

            seconds_for_diff = (
                np.concatenate(
                    [
                        [previous_time],
                        seconds,
                    ]
                )
            )

        else:

            seconds_for_diff = (
                seconds
            )


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
# READ COMPLETE PRESSURE RECORD
# ============================================================

pressure = np.full(
    number_of_samples,
    np.nan,
    dtype=np.float64,
)


invalid_counts = {
    "explicit_fill": 0,
    "integer_sentinel": 0,
    "zero": 0,
    "huge": 0,
    "nonfinite": 0,
}


read_failure_records = []


with h5py.File(file_path, "r") as f:

    pressure_ds = f[
        pressure_variable
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

            raw = pressure_ds[
                start:stop
            ]


            values, counts = (
                decode_pressure_block(
                    raw,
                    pressure_ds,
                )
            )


            pressure[
                start:stop
            ] = values


            for key in invalid_counts:

                invalid_counts[
                    key
                ] += counts[
                    key
                ]


        except Exception as exc:

            read_failure_records.append(
                {
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


# ============================================================
# VALIDITY SUMMARY
# ============================================================

valid = np.isfinite(
    pressure
)

invalid = ~valid


valid_count = int(
    valid.sum()
)


invalid_count = int(
    invalid.sum()
)


valid_fraction = (
    valid_count
    / number_of_samples
)


print("\n" + "=" * 70)
print("PRESSURE DATA VALIDITY")
print("=" * 70)

print(
    "Total samples:",
    number_of_samples,
)

print(
    "Valid samples:",
    valid_count,
)

print(
    "Invalid samples:",
    invalid_count,
)

print(
    f"Valid fraction: "
    f"{valid_fraction:.6f}"
)


print(
    "\nInvalid-value mechanisms:"
)


for key, count in invalid_counts.items():

    print(
        f"  {key:18s}: "
        f"{count}"
    )


# ============================================================
# PRESSURE STATISTICS
# ============================================================

if valid_count > 0:

    valid_pressure = pressure[
        valid
    ]


    (
        minimum,
        q25,
        median,
        q75,
        maximum,
    ) = np.percentile(
        valid_pressure,
        [
            0,
            25,
            50,
            75,
            100,
        ],
    )


    mean_pressure = float(
        np.mean(
            valid_pressure
        )
    )


    std_pressure = float(
        np.std(
            valid_pressure
        )
    )


    print("\n" + "=" * 70)
    print("PRESSURE STATISTICS")
    print("=" * 70)

    print(
        f"Minimum : {minimum:.3f} Pa"
    )

    print(
        f"25%     : {q25:.3f} Pa"
    )

    print(
        f"Median  : {median:.3f} Pa"
    )

    print(
        f"75%     : {q75:.3f} Pa"
    )

    print(
        f"Maximum : {maximum:.3f} Pa"
    )

    print(
        f"Mean    : {mean_pressure:.3f} Pa"
    )

    print(
        f"Std     : {std_pressure:.3f} Pa"
    )


else:

    minimum = np.nan
    q25 = np.nan
    median = np.nan
    q75 = np.nan
    maximum = np.nan
    mean_pressure = np.nan
    std_pressure = np.nan


# ============================================================
# HDF5 READ STATUS
# ============================================================

print("\n" + "=" * 70)
print("HDF5 READ STATUS")
print("=" * 70)


if not read_failure_records:

    print(
        "No HDF5 pressure read failures detected."
    )

else:

    failures = pd.DataFrame(
        read_failure_records
    )

    print(
        "Failed read blocks:",
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
# CONTIGUOUS INVALID-DATA RUNS
# ============================================================

invalid_runs = contiguous_true_runs(
    invalid
)


if not invalid_runs.empty:

    invalid_runs[
        "duration_seconds"
    ] = (
        invalid_runs[
            "sample_count"
        ]
        / detected_fs
    )


    invalid_runs[
        "duration_minutes"
    ] = (
        invalid_runs[
            "duration_seconds"
        ]
        / 60.0
    )


    invalid_runs[
        "duration_hours"
    ] = (
        invalid_runs[
            "duration_seconds"
        ]
        / 3600.0
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


print("\n" + "=" * 70)
print("CONTIGUOUS INVALID-DATA RUNS")
print("=" * 70)

print(
    "Number of invalid runs:",
    len(invalid_runs),
)


if not invalid_runs.empty:

    print(
        invalid_runs.head(
            number_of_gaps_to_report
        ).to_string(
            index=False
        )
    )


# ============================================================
# SAMPLE-TO-SAMPLE VARIABILITY
# ============================================================

paired_valid = (
    valid[:-1]
    & valid[1:]
)


if paired_valid.any():

    dp = (
        pressure[1:]
        - pressure[:-1]
    )


    valid_dp = (
        dp[
            paired_valid
        ]
    )


    unchanged_fraction = float(
        np.mean(
            valid_dp == 0
        )
    )


    median_absolute_change = float(
        np.median(
            np.abs(
                valid_dp
            )
        )
    )


    print("\n" + "=" * 70)
    print("SAMPLE-TO-SAMPLE VARIABILITY")
    print("=" * 70)

    print(
        "Fraction exactly unchanged:",
        unchanged_fraction,
    )

    print(
        "Fraction changed:",
        1.0 - unchanged_fraction,
    )

    print(
        "Median absolute pressure change:",
        median_absolute_change,
        "Pa",
    )


else:

    unchanged_fraction = np.nan


# ============================================================
# 30-MINUTE WINDOW DIAGNOSTICS
# ============================================================

window_samples = int(
    round(
        diagnostic_window_minutes
        * 60.0
        * detected_fs
    )
)


window_records = []


for start in range(
    0,
    number_of_samples,
    window_samples,
):

    stop = min(
        start + window_samples,
        number_of_samples,
    )


    values = pressure[
        start:stop
    ]


    finite = np.isfinite(
        values
    )


    window_valid_fraction = float(
        finite.mean()
    )


    if finite.any():

        window_std = float(
            np.nanstd(
                values
            )
        )


        window_minimum = float(
            np.nanmin(
                values
            )
        )


        window_maximum = float(
            np.nanmax(
                values
            )
        )

    else:

        window_std = np.nan
        window_minimum = np.nan
        window_maximum = np.nan


    all_invalid = (
        finite.sum() == 0
    )


    nearly_frozen = (
        finite.any()
        and
        window_std
        < minimum_window_std_pa
    )


    usable_for_spectrum = (
        window_valid_fraction
        >= minimum_window_valid_fraction
        and not nearly_frozen
    )


    window_records.append(
        {
            "start_index":
                start,
            "stop_index":
                stop,
            "sample_count":
                stop - start,
            "valid_fraction":
                window_valid_fraction,
            "std_pa":
                window_std,
            "minimum_pa":
                window_minimum,
            "maximum_pa":
                window_maximum,
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


usable_window_fraction = float(
    windows[
        "usable_for_spectrum"
    ].mean()
)


print("\n" + "=" * 70)
print("30-MINUTE WINDOW DIAGNOSTICS")
print("=" * 70)

print(
    "Samples per 30-min window:",
    window_samples,
)

print(
    "Number of windows:",
    len(windows),
)

print(
    "All-invalid windows:",
    int(
        windows[
            "all_invalid"
        ].sum()
    ),
)

print(
    "Nearly frozen windows:",
    int(
        windows[
            "nearly_frozen"
        ].sum()
    ),
)

print(
    "Windows passing basic QC:",
    int(
        windows[
            "usable_for_spectrum"
        ].sum()
    ),
)

print(
    f"Fraction usable: "
    f"{usable_window_fraction:.4f}"
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print("\n" + "=" * 70)
print("SURFACE-ELEVATION SPECTRUM ASSESSMENT")
print("=" * 70)


if valid_fraction == 0:

    print(
        "NOT USABLE."
    )

    print(
        "No valid atmospheric-pressure-corrected "
        "pressure samples are available."
    )


elif usable_window_fraction == 0:

    print(
        "NOT USABLE FOR 30-MIN SPECTRA."
    )

    print(
        "No complete 30-minute pressure windows "
        "pass the basic quality criteria."
    )


elif len(read_failure_records) > 0:

    print(
        "PARTIALLY USABLE."
    )

    print(
        "Some parts of the HDF5 pressure dataset "
        "cannot be read."
    )

    print(
        "Use only individual 30-minute windows "
        "that pass quality control."
    )


elif (
    valid_fraction
    < minimum_overall_valid_fraction
    or
    usable_window_fraction < 0.80
):

    print(
        "PARTIALLY USABLE."
    )

    print(
        "Usable periods exist, but burst-level "
        "quality control is essential."
    )


else:

    print(
        "GENERALLY USABLE FOR BURST-BASED "
        "SURFACE-ELEVATION SPECTRAL ANALYSIS."
    )

    print(
        "Individual bursts should still undergo "
        "pressure-response, water-depth, missing-data "
        "and spectral quality checks."
    )