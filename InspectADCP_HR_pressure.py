# -*- coding: utf-8 -*-

"""
Inspect ADCP-HR atmospheric-pressure-corrected pressure (P_APC)
and assess whether it is suitable for surface-elevation
spectral analysis.
"""

import h5py
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADCP-HR"
    r"\pressure_adcp_hr_dvn_201804_F4P3.nc"
)

pressure_variable = "P_APC"

expected_sampling_frequency_hz = 4.0

# Number of samples read at once.
read_block_samples = 500_000

# Spectral-analysis block length.
diagnostic_window_minutes = 30.0

# A submerged pressure sensor should not physically measure
# exactly 0 Pa after atmospheric correction.
treat_zero_as_invalid = True

# Anything beyond this magnitude is regarded as nonphysical /
# sentinel-like.
maximum_abs_pressure_pa = 1.0e7

# Diagnostic threshold for a nearly frozen 30-min record.
minimum_window_std_pa = 1.0

# Minimum valid fraction required within a 30-min burst.
minimum_window_valid_fraction = 0.98

# Used only for the final overall assessment.
minimum_overall_valid_fraction = 0.90

# Number of longest missing-data runs to print.
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

        for attr_name in dset.attrs.keys():

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
    """Return _FillValue and/or missing_value."""

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
    Convert packed P_APC values to Pa and identify invalid data.

    physical pressure =
        raw * scale_factor + add_offset
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

        # For int32 this catches values such as
        # -2147483648 and -2147483647.
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
    # Convert to physical units
    # --------------------------------------------------------

    pressure = (
        raw_float * scale
        + offset
    )


    # --------------------------------------------------------
    # Other suspicious values
    # --------------------------------------------------------

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

    Returns start index, stop index and sample count.
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

    if "Z" not in f:

        raise KeyError(
            "No 'Z' variable found."
        )


    time_ds = f["time"]

    pressure_ds = f[
        pressure_variable
    ]

    number_of_samples = (
        time_ds.shape[0]
    )

    z_m = float(
        np.asarray(
            f["Z"][0]
        ).squeeze()
    )


    if pressure_ds.shape != (
        number_of_samples,
        1,
    ):

        raise ValueError(
            "Expected P_APC shape "
            f"({number_of_samples}, 1), "
            f"found {pressure_ds.shape}."
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


print("\n" + "=" * 70)
print("FILE STRUCTURE")
print("=" * 70)

print(
    "Number of samples:",
    number_of_samples,
)

print(
    "Pressure shape:",
    (
        number_of_samples,
        1,
    ),
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
    "Instrument height Z:",
    z_m,
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


    # --------------------------------------------------------
    # Start / end time
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Read time in blocks
    # --------------------------------------------------------

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

        raw_time = (
            time_ds[
                start:stop
            ]
            .astype(np.float64)
        )

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
# READ COMPLETE P_APC RECORD
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


        print(
            f"Reading pressure "
            f"{start}:{stop}"
        )


        try:

            raw = pressure_ds[
                start:stop,
                0,
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
print("P_APC DATA VALIDITY")
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


    percentiles = np.percentile(
        valid_pressure,
        [
            0,
            25,
            50,
            75,
            100,
        ],
    )


    minimum = percentiles[0]
    q25 = percentiles[1]
    median = percentiles[2]
    q75 = percentiles[3]
    maximum = percentiles[4]


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
    print("P_APC PRESSURE STATISTICS")
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
# HDF5 READ FAILURES
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
        "Number of failed read blocks:",
        len(failures),
    )

    print(
        failures.to_string(
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
        "\nLongest invalid runs:"
    )

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


    valid_dp = dp[
        paired_valid
    ]


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


    valid_fraction_window = float(
        finite.mean()
    )


    if finite.any():

        std_window = float(
            np.nanstd(
                values
            )
        )

        minimum_window = float(
            np.nanmin(
                values
            )
        )

        maximum_window = float(
            np.nanmax(
                values
            )
        )

    else:

        std_window = np.nan
        minimum_window = np.nan
        maximum_window = np.nan


    all_invalid = (
        finite.sum() == 0
    )


    nearly_frozen = (
        finite.any()
        and
        std_window
        < minimum_window_std_pa
    )


    usable_for_spectrum = (
        valid_fraction_window
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
                valid_fraction_window,
            "std_pa":
                std_window,
            "minimum_pa":
                minimum_window,
            "maximum_pa":
                maximum_window,
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
    f"Fraction of windows usable: "
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
        "P_APC contains no valid corrected-pressure samples."
    )


elif len(read_failure_records) > 0:

    print(
        "PARTIALLY USABLE."
    )

    print(
        "Some HDF5 blocks cannot be read."
    )

    print(
        "Only complete 30-minute windows that pass QC "
        "should be used for spectra."
    )


elif usable_window_fraction < 0.50:

    print(
        "POOR FOR SPECTRAL ANALYSIS."
    )

    print(
        "Less than half of the 30-minute windows "
        "pass the basic pressure QC."
    )


elif (
    valid_fraction
    < minimum_overall_valid_fraction
    or usable_window_fraction < 0.80
):

    print(
        "PARTIALLY USABLE."
    )

    print(
        "There are usable periods, but burst-level QC "
        "is essential."
    )


else:

    print(
        "GENERALLY USABLE FOR BURST-BASED "
        "SURFACE-ELEVATION SPECTRAL ANALYSIS."
    )

    print(
        "Individual 30-minute bursts should still undergo "
        "missing-data, frozen-signal, water-depth and "
        "pressure-response-factor QC."
    )