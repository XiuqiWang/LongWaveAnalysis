# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:48:39 2026

@author: WangX3

Read atmospheric-pressure-corrected pressure from a KG2 pressure
NetCDF file and save one ADV pressure record as Parquet.

The file dimensions are:

    P_APC(time, instrument)

Column 0 corresponds to the first instrument listed in "instrument".
Column 1 corresponds to the second instrument.

Unreadable HDF5 blocks are handled by stopping at the first corrupted
block and retaining the continuous data that were read successfully.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# FUNCTIONS
# ============================================================

def _scalar_attribute(dataset, name, default):
    """
    Read a scalar NetCDF/HDF5 attribute as a Python float.
    """
    value = dataset.attrs.get(name, default)

    return float(
        np.asarray(value).squeeze()
    )


def _read_pressure_chunked(
    dataset,
    start_index,
    stop_index,
    column,
    block_samples=100_000,
):
    """
    Read one pressure column in blocks.

    The NetCDF scale_factor and add_offset are applied automatically.

    Reading stops at the first unreadable HDF5 block. Only the
    successfully read continuous portion is returned.

    Parameters
    ----------
    dataset : h5py.Dataset
        The P_APC dataset.
    start_index, stop_index : int
        Requested sample-index range.
    column : int
        Instrument column: 0 for the first instrument and 1 for
        the second.
    block_samples : int
        Number of samples attempted in each read operation.

    Returns
    -------
    pressure_pa : numpy.ndarray
        Atmospheric-pressure-corrected water pressure in Pa.
    failures : list[dict]
        Information about unreadable HDF5 blocks.
    """
    number_of_samples = stop_index - start_index

    output = np.full(
        number_of_samples,
        np.nan,
        dtype=np.float64,
    )

    failures = []

    scale = _scalar_attribute(
        dataset,
        "scale_factor",
        1.0,
    )

    offset = _scalar_attribute(
        dataset,
        "add_offset",
        0.0,
    )

    fill_value = dataset.attrs.get(
        "_FillValue",
        None,
    )

    if fill_value is not None:
        fill_value = float(
            np.asarray(fill_value).squeeze()
        )

    for block_start in range(
        start_index,
        stop_index,
        block_samples,
    ):
        block_stop = min(
            block_start + block_samples,
            stop_index,
        )

        destination_start = (
            block_start - start_index
        )

        destination_stop = (
            block_stop - start_index
        )

        try:
            raw = dataset[
                block_start:block_stop,
                column,
            ].astype(np.float64)

            if fill_value is not None:
                raw[raw == fill_value] = np.nan

            output[
                destination_start:destination_stop
            ] = raw * scale + offset

        except OSError as exc:
            failures.append(
                {
                    "variable": dataset.name,
                    "column": column,
                    "start_index": block_start,
                    "stop_index": block_stop,
                    "error": str(exc),
                }
            )

            print(
                "\nUnreadable pressure block encountered:"
            )
            print(
                block_start,
                "to",
                block_stop,
            )
            print(type(exc).__name__, exc)

            # Retain only the continuous part before corruption.
            return (
                output[:destination_start],
                failures,
            )

    return output, failures


def _read_time_chunked(
    dataset,
    start_index,
    stop_index,
    block_samples=100_000,
):
    """
    Read and decode the time coordinate in blocks.

    Reading stops at the first unreadable HDF5 block.
    """
    number_of_samples = stop_index - start_index

    output = np.full(
        number_of_samples,
        np.nan,
        dtype=np.float64,
    )

    failures = []

    scale = _scalar_attribute(
        dataset,
        "scale_factor",
        1.0,
    )

    offset = _scalar_attribute(
        dataset,
        "add_offset",
        0.0,
    )

    for block_start in range(
        start_index,
        stop_index,
        block_samples,
    ):
        block_stop = min(
            block_start + block_samples,
            stop_index,
        )

        destination_start = (
            block_start - start_index
        )

        destination_stop = (
            block_stop - start_index
        )

        try:
            raw_time = dataset[
                block_start:block_stop
            ].astype(np.float64)

            output[
                destination_start:destination_stop
            ] = raw_time * scale + offset

        except OSError as exc:
            failures.append(
                {
                    "variable": "time",
                    "start_index": block_start,
                    "stop_index": block_stop,
                    "error": str(exc),
                }
            )

            print(
                "\nUnreadable time block encountered:"
            )
            print(
                block_start,
                "to",
                block_stop,
            )
            print(type(exc).__name__, exc)

            return (
                output[:destination_start],
                failures,
            )

    return output, failures


def find_time_index(
    time_ds,
    target_time,
):
    """
    Find the first NetCDF sample at or after target_time.

    The complete time variable is not loaded into memory.
    """
    target_time = pd.Timestamp(target_time)

    if target_time.tzinfo is None:
        target_time = target_time.tz_localize(
            "UTC"
        )
    else:
        target_time = target_time.tz_convert(
            "UTC"
        )

    scale = _scalar_attribute(
        time_ds,
        "scale_factor",
        1.0,
    )

    offset = _scalar_attribute(
        time_ds,
        "add_offset",
        0.0,
    )

    target_raw = (
        target_time.timestamp() - offset
    ) / scale

    low = 0
    high = time_ds.shape[0]

    while low < high:
        middle = (low + high) // 2

        try:
            middle_value = float(
                time_ds[middle]
            )
        except OSError as exc:
            raise OSError(
                "Could not read the time coordinate while "
                "searching for the requested timestamp."
            ) from exc

        if middle_value < target_raw:
            low = middle + 1
        else:
            high = middle

    return low


def read_pressure_block_chunked(
    file_path,
    pressure_column,
    instrument_name,
    z_pressure_m,
    start_index,
    stop_index,
    block_samples=100_000,
):
    """
    Read one P_APC instrument column from a KG2 pressure file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        KG2 pressure NetCDF file.
    pressure_column : int
        Instrument column in P_APC.
    instrument_name : str
        Instrument label, for example "ADV01".
    z_pressure_m : float
        Pressure-sensor height above the frame/seabed.
    start_index, stop_index : int
        Sample-index limits.
    block_samples : int
        Samples attempted in each HDF5 read.

    Returns
    -------
    result : pandas.DataFrame
        Time and atmospheric-pressure-corrected pressure.
    failures : list[dict]
        Information about unreadable HDF5 blocks.
    """
    file_path = Path(file_path)

    height_cm = round(
        z_pressure_m * 100
    )

    all_failures = []

    with h5py.File(file_path, "r") as f:
        required_variables = {
            "time",
            "P_APC",
            "instrument",
            "Z_pres",
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
        pressure_ds = f["P_APC"]

        number_of_samples = time_ds.shape[0]

        if pressure_ds.shape[0] != number_of_samples:
            raise ValueError(
                "The time and P_APC dimensions differ."
            )

        if pressure_column < 0:
            raise ValueError(
                "pressure_column cannot be negative."
            )

        if pressure_column >= pressure_ds.shape[1]:
            raise IndexError(
                f"pressure_column={pressure_column} is invalid. "
                f"P_APC has {pressure_ds.shape[1]} columns."
            )

        start_index = max(
            0,
            int(start_index),
        )

        stop_index = min(
            int(stop_index),
            number_of_samples,
        )

        if stop_index <= start_index:
            raise ValueError(
                "stop_index must be greater than start_index."
            )

        epoch_seconds, time_failures = (
            _read_time_chunked(
                dataset=time_ds,
                start_index=start_index,
                stop_index=stop_index,
                block_samples=block_samples,
            )
        )

        for failure in time_failures:
            failure.update(
                {
                    "instrument": instrument_name,
                    "column": pressure_column,
                }
            )
            all_failures.append(failure)

        pressure_pa, pressure_failures = (
            _read_pressure_chunked(
                dataset=pressure_ds,
                start_index=start_index,
                stop_index=stop_index,
                column=pressure_column,
                block_samples=block_samples,
            )
        )

        for failure in pressure_failures:
            failure.update(
                {
                    "instrument": instrument_name,
                    "column": pressure_column,
                }
            )
            all_failures.append(failure)

        # Time and pressure may stop at different locations if one
        # variable encounters an unreadable block first.
        successful_length = min(
            len(epoch_seconds),
            len(pressure_pa),
        )

        epoch_seconds = (
            epoch_seconds[:successful_length]
        )

        pressure_pa = (
            pressure_pa[:successful_length]
        )

        pressure_column_name = (
            f"pressure_apc_"
            f"{instrument_name}_"
            f"{height_cm}cm_pa"
        )

        result = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    epoch_seconds,
                    unit="s",
                    utc=True,
                    errors="coerce",
                ),
                pressure_column_name:
                    pressure_pa,
            }
        )

    return result, all_failures


def add_failure_times(
    failures,
    file_path,
):
    """
    Add approximate UTC times to failure records.
    """
    if not failures:
        return failures

    with h5py.File(file_path, "r") as f:
        time_ds = f["time"]

        scale = _scalar_attribute(
            time_ds,
            "scale_factor",
            1.0,
        )

        offset = _scalar_attribute(
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

            except OSError:
                failure["start_time_utc"] = (
                    pd.NaT
                )
                failure["end_time_utc"] = (
                    pd.NaT
                )

    return failures


# ============================================================
# USER SETTINGS
# ============================================================

file_path = Path(
    r"C:\dev\Python\LongWaveAnalysis\ADV"
    r"\pressure_adv_dvn_201804_F3.nc"
)

# P_APC column:
#     0 = ADV01
#     1 = ADV02
pressure_column = 0

instrument_name = "ADV05"

# From Z_pres:
#     ADV01 = 0.895 m
#     ADV02 = 0.596 m
z_pressure_m = 0.90

# Select the time period to extract.
#
# Replace these with the actual deployment limits that you want
# to analyze.
target_time_start = pd.Timestamp(
    "2018-04-04 00:00:00",
    tz="UTC",
)

target_time_stop = pd.Timestamp(
    "2018-05-15 17:00:00",
    tz="UTC",
)

block_samples = 100_000


# ============================================================
# INSPECT INSTRUMENT METADATA
# ============================================================

with h5py.File(file_path, "r") as f:
    instruments = [
        value.decode()
        if isinstance(value, bytes)
        else str(value)
        for value in f["instrument"][...]
    ]

    pressure_heights = (
        f["Z_pres"][...]
        .astype(float)
    )

    print("Instruments:", instruments)
    print(
        "Pressure-sensor heights:",
        pressure_heights,
        "m",
    )

    file_instrument = (
        instruments[pressure_column]
    )

    file_height = float(
        pressure_heights[pressure_column]
    )

    print(
        "\nSelected NetCDF column:",
        pressure_column,
    )
    print(
        "Instrument in file:",
        file_instrument,
    )
    print(
        "Pressure height in file:",
        file_height,
        "m",
    )

    if file_instrument != instrument_name:
        raise ValueError(
            f"Column {pressure_column} contains "
            f"{file_instrument}, not {instrument_name}."
        )

    if not np.isclose(
        file_height,
        z_pressure_m,
        atol=0.001,
    ):
        raise ValueError(
            "The selected z_pressure_m does not match "
            f"Z_pres in the file: {file_height} m."
        )


# ============================================================
# FIND EXTRACTION INDICES
# ============================================================

with h5py.File(file_path, "r") as f:
    start_index = find_time_index(
        time_ds=f["time"],
        target_time=target_time_start,
    )

    stop_index = find_time_index(
        time_ds=f["time"],
        target_time=target_time_stop,
    )

print("\nRequested start index:", start_index)
print("Requested stop index: ", stop_index)


# ============================================================
# READ P_APC
# ============================================================

df, failures = read_pressure_block_chunked(
    file_path=file_path,
    pressure_column=pressure_column,
    instrument_name=instrument_name,
    z_pressure_m=z_pressure_m,
    start_index=start_index,
    stop_index=stop_index,
    block_samples=block_samples,
)


# ============================================================
# QUALITY CHECKS
# ============================================================

pressure_name = (
    f"pressure_apc_"
    f"{instrument_name}_"
    f"{round(z_pressure_m * 100)}cm_pa"
)

if df.empty:
    raise RuntimeError(
        "No pressure data were read."
    )

print(
    "\nExtracted",
    instrument_name,
    "pressure data",
)

print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows: ", len(df))

print("\nMissing values:")
print(df.isna().sum())

print("\nPressure summary in Pa:")
print(
    df[pressure_name].describe()
)

# Eliminate rows with an invalid timestamp.
df = (
    df.loc[df["time"].notna()]
    .reset_index(drop=True)
)

# Do not automatically remove missing pressure values here.
# Keeping them makes data problems visible during later processing.

if failures:
    failures = add_failure_times(
        failures=failures,
        file_path=file_path,
    )

    print("\nUnreadable blocks:")

    for failure in failures:
        print(
            failure["variable"],
            failure["instrument"],
            failure.get(
                "start_time_utc",
                pd.NaT,
            ),
            "to",
            failure.get(
                "end_time_utc",
                pd.NaT,
            ),
        )
else:
    print("\nNo unreadable blocks encountered.")


# ============================================================
# SAVE PROCESSED PRESSURE DATA
# ============================================================

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    output_folder
    / "Pressure_F3_ADV05_APC.parquet"
)

df.to_parquet(
    output_file,
    index=False,
)

print("\nSaved processed pressure data to:")
print(output_file)