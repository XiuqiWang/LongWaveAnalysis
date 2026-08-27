# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:47:17 2026

@author: WangX3

Read corrected pressure from a KG2 ADCP pressure NetCDF file
and save it to Parquet.

Expected pressure variable:
    pressure_corrected

The file contains one pressure sensor.

The sensor depth is read from the metadata of pressure_corrected.
The code first looks for a direct 'sensor_depth' attribute. If it
is not present, it attempts to extract sensor_depth from the
'description' attribute.

Unreadable HDF5 blocks remain NaN so later data can still be read.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

file_path = Path(
    r"C:\dev\Python\LongWaveAnalysis\ADCP"
    r"\UP\adcp_dvn_201804_f3p3_000.nc"
)

case_id = "DVN_F3_ADCP"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)
output_folder.mkdir(parents=True, exist_ok=True)

output_file = (
    output_folder
    / f"Pressure_{case_id}.parquet"
)

# metadata_file = (
#     output_folder
#     / f"Pressure_{case_id}_metadata.csv"
# )

# failure_file = (
#     output_folder
#     / f"Pressure_{case_id}_read_failures.csv"
# )

# Set either to None to read the complete available record.
target_time_start = pd.Timestamp(
    "2018-04-04 13:00:00",
    tz="UTC",
)

target_time_stop = pd.Timestamp(
    "2018-05-18 07:00:00",
    tz="UTC",
)

block_samples = 100_000


# ============================================================
# HELPERS
# ============================================================

def scalar_attribute(dataset, name, default=1.0):
    """Return scalar NetCDF/HDF5 attribute as float."""
    value = dataset.attrs.get(name, default)
    return float(np.asarray(value).squeeze())


def dataset_fill_value(dataset):
    """Return _FillValue or missing_value, if available."""
    value = dataset.attrs.get("_FillValue", None)

    if value is None:
        value = dataset.attrs.get("missing_value", None)

    if value is None:
        return None

    return float(np.asarray(value).squeeze())


def decode_attribute(value):
    """Convert an HDF5 attribute to readable text."""
    if value is None:
        return ""

    value = np.asarray(value).squeeze()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def read_sensor_depth(pressure_ds):
    """
    Read sensor depth from pressure_corrected metadata.

    Priority:
    1. direct 'sensor_depth' attribute;
    2. extract sensor_depth from description text.
    """

    if "sensor_depth" in pressure_ds.attrs:
        value = pressure_ds.attrs["sensor_depth"]

        try:
            return float(np.asarray(value).squeeze())
        except (TypeError, ValueError):
            pass

    description = decode_attribute(
        pressure_ds.attrs.get("description", "")
    )

    # Examples this can match:
    # sensor_depth = 0.42
    # sensor_depth: 0.42 m
    # sensor depth = 0.42 m
    pattern = re.compile(
        r"sensor[_\s-]*depth"
        r"\s*[:=]?\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*(?:m|meter|metres|meters)?",
        re.IGNORECASE,
    )

    match = pattern.search(description)

    if match:
        return float(match.group(1))

    raise ValueError(
        "Could not determine sensor_depth from the "
        "pressure_corrected metadata.\n"
        f"description = {description!r}"
    )


def find_time_index(time_ds, target_time):
    """
    Return first sample index at or after target_time.

    Does not load the entire time coordinate into memory.
    """

    target_time = pd.Timestamp(target_time)

    if target_time.tzinfo is None:
        target_time = target_time.tz_localize("UTC")
    else:
        target_time = target_time.tz_convert("UTC")

    scale = scalar_attribute(
        time_ds,
        "scale_factor",
        1.0,
    )

    offset = scalar_attribute(
        time_ds,
        "add_offset",
        0.0,
    )

    target_raw = (
        target_time.timestamp()
        - offset
    ) / scale

    low = 0
    high = time_ds.shape[0]

    while low < high:
        middle = (low + high) // 2
        middle_value = float(time_ds[middle])

        if middle_value < target_raw:
            low = middle + 1
        else:
            high = middle

    return low


def read_time_chunked(
    dataset,
    start,
    stop,
    block_samples,
):
    """Read time coordinate in blocks."""

    output = np.full(
        stop - start,
        np.nan,
        dtype=np.float64,
    )

    failures = []

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

    fill_value = dataset_fill_value(dataset)

    for block_start in range(
        start,
        stop,
        block_samples,
    ):
        block_stop = min(
            block_start + block_samples,
            stop,
        )

        j0 = block_start - start
        j1 = block_stop - start

        try:
            raw = dataset[
                block_start:block_stop
            ].astype(np.float64)

            if fill_value is not None:
                raw[raw == fill_value] = np.nan

            output[j0:j1] = (
                raw * scale + offset
            )

        except (OSError, RuntimeError) as exc:
            failures.append({
                "variable": "time",
                "start_index": block_start,
                "stop_index": block_stop,
                "error": str(exc),
            })

    return output, failures


def read_pressure_chunked(
    dataset,
    start,
    stop,
    block_samples,
):
    """
    Read pressure_corrected in blocks.

    Supports:
        pressure_corrected(time)
        pressure_corrected(time, 1)
    """

    output = np.full(
        stop - start,
        np.nan,
        dtype=np.float64,
    )

    failures = []

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

    fill_value = dataset_fill_value(dataset)

    if dataset.ndim not in (1, 2):
        raise ValueError(
            "pressure_corrected must have shape "
            "(time,) or (time, 1). "
            f"Found {dataset.shape}."
        )

    if dataset.ndim == 2 and dataset.shape[1] != 1:
        raise ValueError(
            "Expected only one pressure sensor, but "
            f"pressure_corrected has shape {dataset.shape}."
        )

    for block_start in range(
        start,
        stop,
        block_samples,
    ):
        block_stop = min(
            block_start + block_samples,
            stop,
        )

        j0 = block_start - start
        j1 = block_stop - start

        try:
            if dataset.ndim == 1:
                raw = dataset[
                    block_start:block_stop
                ].astype(np.float64)

            else:
                raw = dataset[
                    block_start:block_stop,
                    0,
                ].astype(np.float64)

            if fill_value is not None:
                raw[raw == fill_value] = np.nan

            output[j0:j1] = (
                raw * scale + offset
            )

        except (OSError, RuntimeError) as exc:
            failures.append({
                "variable": "pressure_corrected",
                "start_index": block_start,
                "stop_index": block_stop,
                "error": str(exc),
            })

    return output, failures


def add_failure_times(
    failures,
    file_path,
):
    """Add approximate UTC times to failed block records."""

    if not failures:
        return failures

    with h5py.File(file_path, "r") as f:
        time_ds = f["time"]

        scale = scalar_attribute(
            time_ds,
            "scale_factor",
            1.0,
        )

        offset = scalar_attribute(
            time_ds,
            "add_offset",
            0.0,
        )

        for failure in failures:
            i0 = failure["start_index"]

            i1 = min(
                failure["stop_index"] - 1,
                time_ds.shape[0] - 1,
            )

            try:
                t0 = (
                    float(time_ds[i0])
                    * scale
                    + offset
                )

                t1 = (
                    float(time_ds[i1])
                    * scale
                    + offset
                )

                failure["start_time_utc"] = (
                    pd.to_datetime(
                        t0,
                        unit="s",
                        utc=True,
                    )
                )

                failure["end_time_utc"] = (
                    pd.to_datetime(
                        t1,
                        unit="s",
                        utc=True,
                    )
                )

            except (OSError, RuntimeError):
                failure["start_time_utc"] = pd.NaT
                failure["end_time_utc"] = pd.NaT

    return failures


# ============================================================
# INSPECT FILE + DETERMINE EXTRACTION RANGE
# ============================================================

with h5py.File(file_path, "r") as f:

    required_variables = {
        "time",
        "pressure_corrected",
    }

    missing = (
        required_variables
        - set(f.keys())
    )

    if missing:
        raise KeyError(
            "Missing required variables: "
            f"{sorted(missing)}"
        )

    time_ds = f["time"]
    pressure_ds = f["pressure_corrected"]

    if pressure_ds.shape[0] != time_ds.shape[0]:
        raise ValueError(
            "time and pressure_corrected "
            "have different sample counts."
        )

    sensor_depth_m = read_sensor_depth(
        pressure_ds
    )

    number_of_samples_total = (
        time_ds.shape[0]
    )

    start_index = (
        0
        if target_time_start is None
        else find_time_index(
            time_ds,
            target_time_start,
        )
    )

    stop_index = (
        number_of_samples_total
        if target_time_stop is None
        else find_time_index(
            time_ds,
            target_time_stop,
        )
    )

    start_index = max(
        0,
        int(start_index),
    )

    stop_index = min(
        number_of_samples_total,
        int(stop_index),
    )

    if stop_index <= start_index:
        raise ValueError(
            "Selected time interval contains no samples."
        )

    pressure_units = decode_attribute(
        pressure_ds.attrs.get(
            "units",
            "",
        )
    )

    pressure_description = decode_attribute(
        pressure_ds.attrs.get(
            "description",
            "",
        )
    )

    pressure_long_name = decode_attribute(
        pressure_ds.attrs.get(
            "long_name",
            "",
        )
    )

    print("pressure_corrected shape:", pressure_ds.shape)
    print("pressure_corrected units:", pressure_units)
    print("Sensor depth:", sensor_depth_m, "m")

    print("\nDescription:")
    print(pressure_description)


# ============================================================
# READ DATA
# ============================================================

all_failures = []

with h5py.File(file_path, "r") as f:

    epoch_seconds, failures = read_time_chunked(
        dataset=f["time"],
        start=start_index,
        stop=stop_index,
        block_samples=block_samples,
    )

    all_failures.extend(failures)

    pressure, failures = read_pressure_chunked(
        dataset=f["pressure_corrected"],
        start=start_index,
        stop=stop_index,
        block_samples=block_samples,
    )

    all_failures.extend(failures)


# ============================================================
# BUILD OUTPUT TABLE
# ============================================================

# Keep one simple, stable pressure name.
# Units are preserved separately in metadata.
result = pd.DataFrame({
    "time": pd.to_datetime(
        epoch_seconds,
        unit="s",
        utc=True,
        errors="coerce",
    ),
    "pressure_corrected": pressure,
})

# Rows with invalid time cannot be aligned later.
missing_time_count = (
    result["time"].isna().sum()
)

if missing_time_count:
    print(
        "\nDropping",
        missing_time_count,
        "rows with invalid timestamps.",
    )

    result = (
        result.dropna(
            subset=["time"]
        )
        .reset_index(drop=True)
    )

if result.empty:
    raise RuntimeError(
        "No pressure samples remain after reading."
    )


# ============================================================
# QUALITY CHECKS
# ============================================================

print("\nExtracted ADCP pressure data")
print("Case:", case_id)
print("Start:", result["time"].iloc[0])
print("End:  ", result["time"].iloc[-1])
print("Rows: ", len(result))

print("\nMissing values:")
print(result.isna().sum())

print("\nCorrected pressure summary:")
print(
    result["pressure_corrected"]
    .describe()
)


# ============================================================
# METADATA TABLE
# ============================================================

metadata = pd.DataFrame({
    "case_id": [case_id],
    "pressure_variable": [
        "pressure_corrected"
    ],
    "sensor_depth_m": [
        sensor_depth_m
    ],
    "pressure_units": [
        pressure_units
    ],
    "pressure_long_name": [
        pressure_long_name
    ],
    "pressure_description": [
        pressure_description
    ],
    "source_file": [
        str(file_path)
    ],
    "start_time_utc": [
        result["time"].iloc[0]
    ],
    "end_time_utc": [
        result["time"].iloc[-1]
    ],
    "sample_count": [
        len(result)
    ],
})


# ============================================================
# FAILURE INFORMATION
# ============================================================

if all_failures:
    all_failures = add_failure_times(
        failures=all_failures,
        file_path=file_path,
    )

    # pd.DataFrame(
    #     all_failures
    # ).to_csv(
    #     failure_file,
    #     index=False,
    # )

    # print(
    #     "\nRead failures saved to:"
    # )
    # print(failure_file)

    print(
        "Number of failed blocks:",
        len(all_failures),
    )

else:
    print(
        "\nNo HDF5 read failures detected."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

result.to_parquet(
    output_file,
    index=False,
)

# metadata.to_csv(
#     metadata_file,
#     index=False,
# )

print(
    "\nSaved corrected ADCP pressure to:"
)
print(output_file)

# print(
#     "\nSaved pressure metadata to:"
# )
# print(metadata_file)