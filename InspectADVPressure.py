# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:20:26 2026

@author: WangX3
"""

import h5py
import numpy as np
import pandas as pd

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADV"
    r"\pressure_adv_dvn_201804_F1.nc"
)

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
            print("Could not inspect dataset:", type(exc).__name__, exc)
            continue

        print("Attributes:")

        try:
            attr_names = list(dset.attrs.keys())
        except Exception as exc:
            print(
                "Could not list attributes:",
                type(exc).__name__,
                exc,
            )
            attr_names = []

        for attr_name in attr_names:
            try:
                print(" ", attr_name, "=", dset.attrs[attr_name])
            except Exception as exc:
                print(
                    " ",
                    attr_name,
                    "FAILED:",
                    type(exc).__name__,
                    exc,
                )

# # inspects the second dimension -> two elevations
# names = [
#     "instrument",
#     "lat",
#     "lon",
#     "Z",
#     "Z_vel",
#     "Z_pres",
# ]

# with h5py.File(file_path, "r") as f:
#     for name in names:
#         print("\n" + "=" * 60)
#         print(name)

#         dset = f[name]
#         print("shape:", dset.shape)
#         print("dtype:", dset.dtype)

#         try:
#             print("values:", dset[...])
#         except Exception as exc:
#             print("Could not read values:", type(exc).__name__, exc)

#         try:
#             print("attributes:")
#             for attr_name in dset.attrs.keys():
#                 print(" ", attr_name, "=", dset.attrs[attr_name])
#         except Exception as exc:
#             print("Could not read attributes:", type(exc).__name__, exc)

# ============================================================
# HELPER FOR SCALE / OFFSET
# ============================================================

def scalar_attr(dataset, name, default):
    value = dataset.attrs.get(name, default)
    return float(np.asarray(value).squeeze())


# ============================================================
# INSPECT TIME SAMPLING
# ============================================================

with h5py.File(file_path, "r") as f:

    if "time" not in f:
        raise KeyError("No 'time' variable found.")

    time_ds = f["time"]

    time_scale = scalar_attr(
        time_ds,
        "scale_factor",
        1.0,
    )

    time_offset = scalar_attr(
        time_ds,
        "add_offset",
        0.0,
    )

    # Read a manageable number of samples for diagnostics.
    n_test = min(
        200_000,
        time_ds.shape[0],
    )

    raw_time = (
        time_ds[:n_test]
        .astype(np.float64)
    )

    epoch_seconds = (
        raw_time * time_scale
        + time_offset
    )

    dt = np.diff(epoch_seconds)

    positive_dt = dt[
        np.isfinite(dt)
        & (dt > 0)
    ]

    if positive_dt.size == 0:
        raise RuntimeError(
            "No positive timestamp intervals found."
        )

    median_dt = np.nanmedian(
        positive_dt
    )

    fs_detected = (
        1.0 / median_dt
    )

    print("\n" + "=" * 70)
    print("TIME SAMPLING CHECK")
    print("=" * 70)

    print(
        "Number of time samples checked:",
        n_test,
    )

    print(
        "Median positive dt:",
        median_dt,
        "s",
    )

    print(
        "Detected sampling frequency:",
        fs_detected,
        "Hz",
    )

    print(
        "Smallest dt:",
        np.nanmin(positive_dt),
        "s",
    )

    print(
        "Largest dt:",
        np.nanmax(positive_dt),
        "s",
    )

    # Show the most common timestamp intervals.
    rounded_dt = np.round(
        positive_dt,
        6,
    )

    unique_dt, counts_dt = np.unique(
        rounded_dt,
        return_counts=True,
    )

    order = np.argsort(
        counts_dt
    )[::-1]

    print("\nMost common dt values:")

    for idx in order[:10]:
        print(
            f"  dt = {unique_dt[idx]:.6f} s : "
            f"{counts_dt[idx]} occurrences"
        )

with h5py.File(file_path, "r") as f:
    pressure_ds = f["P_APC"]
    pressure_scale = scalar_attr(
            pressure_ds,
            "scale_factor",
            1.0,
        )
    
    pressure_offset = scalar_attr(
        pressure_ds,
        "add_offset",
        0.0,
    )

    n_test = min(
        200_000,
        pressure_ds.shape[0],
        time_ds.shape[0],
    )
    
    raw_pressure = (
        pressure_ds[:n_test]
        .astype(np.float64)
    )
    
    # If pressure is 2-D, inspect first column.
    if raw_pressure.ndim == 2:
        print(
            "Pressure is 2-D; using column 0 "
            "for this diagnostic."
        )

        raw_pressure = (
            raw_pressure[:, 0]
        )
    
    pressure = (
            raw_pressure * pressure_scale
            + pressure_offset
        )
    
    # Handle fill value if present.
    fill_value = pressure_ds.attrs.get(
        "_FillValue",
        None,
    )
    
    if fill_value is not None:
        fill_value = float(
            np.asarray(fill_value).squeeze()
        )
    
        pressure[
            raw_pressure == fill_value
        ] = np.nan
    
# ============================================================
# CHECK WHETHER PRESSURE CHANGES EACH SAMPLE
# ============================================================

finite_pair = (
    np.isfinite(pressure[:-1])
    & np.isfinite(pressure[1:])
)

dp = (
    pressure[1:]
    - pressure[:-1]
)

valid_dp = dp[
    finite_pair
]

unchanged_fraction = np.mean(
    valid_dp == 0
)

changed_fraction = np.mean(
    valid_dp != 0
)

print("\n" + "=" * 70)
print("PRESSURE SAMPLE-BY-SAMPLE CHECK")
print("=" * 70)

print(
    "Pressure samples checked:",
    len(pressure),
)

print(
    "Fraction unchanged from previous sample:",
    unchanged_fraction,
)

print(
    "Fraction changed from previous sample:",
    changed_fraction,
)

print(
    "Median absolute change per stored sample:",
    np.nanmedian(
        np.abs(valid_dp)
    ),
)

print(
    "Mean absolute change per stored sample:",
    np.nanmean(
        np.abs(valid_dp)
    ),
)

# ============================================================
# RUN LENGTHS OF IDENTICAL VALUES
# ============================================================

pressure_series = pd.Series(
    pressure
)

change_group = (
    pressure_series
    .ne(
        pressure_series.shift()
    )
    .cumsum()
)

run_lengths = (
    pressure_series
    .groupby(change_group)
    .size()
)

print("\nRun-length distribution of identical values:")
print(
    run_lengths
    .value_counts()
    .sort_index()
    .head(30)
)

print("\nRun-length statistics:")
print(
    run_lengths.describe()
)

# ============================================================
# PRINT FIRST 40 TIME / PRESSURE PAIRS
# ============================================================

with h5py.File(file_path, "r") as f:

    time_ds = f["time"]

    raw_time = (
        time_ds[:40]
        .astype(np.float64)
    )

    time_scale = scalar_attr(
        time_ds,
        "scale_factor",
        1.0,
    )

    time_offset = scalar_attr(
        time_ds,
        "add_offset",
        0.0,
    )

    epoch_seconds = (
        raw_time * time_scale
        + time_offset
    )

    timestamps = pd.to_datetime(
        epoch_seconds,
        unit="s",
        utc=True,
    )

inspection = pd.DataFrame(
    {
        "time": timestamps,
        "pressure": pressure[:40],
    }
)

inspection["changed_from_previous"] = (
    inspection["pressure"]
    .diff()
    .ne(0)
)

print("\nFirst 40 samples:")
print(
    inspection.to_string(
        index=False
    )
)

# ============================================================
# OPTIONAL: CHECK EVERY-4TH-SAMPLE STRUCTURE
# ============================================================

if len(pressure) >= 8:

    p0 = pressure[0::4]
    p1 = pressure[1::4]
    p2 = pressure[2::4]
    p3 = pressure[3::4]

    n = min(
        len(p0),
        len(p1),
        len(p2),
        len(p3),
    )

    equal_01 = np.mean(
        p0[:n] == p1[:n]
    )

    equal_02 = np.mean(
        p0[:n] == p2[:n]
    )

    equal_03 = np.mean(
        p0[:n] == p3[:n]
    )

    print("\n" + "=" * 70)
    print("CHECK FOR 4-HZ DATA REPEATED ON 16-HZ GRID")
    print("=" * 70)

    print(
        "Fraction sample 0 == sample 1 within each group of 4:",
        equal_01,
    )

    print(
        "Fraction sample 0 == sample 2 within each group of 4:",
        equal_02,
    )

    print(
        "Fraction sample 0 == sample 3 within each group of 4:",
        equal_03,
    )

    print(
        "\nIf the pressure were simply one 4-Hz value repeated "
        "four times on a 16-Hz timeline, these three fractions "
        "should all be close to 1."
    )
