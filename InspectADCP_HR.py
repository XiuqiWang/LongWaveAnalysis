# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:18:13 2026

@author: WangX3
"""

import h5py
import numpy as np

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADCP-HR"
    r"\adcp_hr_dvn_201804_F1P1.nc"
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

# ============================================================
# CHECK EAST/NORTH DATA AVAILABILITY FOR ALL 13 CELLS
# ============================================================

import pandas as pd


def scalar_attribute(dataset, name, default):
    """Return a scalar HDF5 attribute."""
    return float(
        np.asarray(
            dataset.attrs.get(name, default)
        ).squeeze()
    )


def get_fill_value(dataset):
    """Return _FillValue or missing_value, if available."""
    fill_value = dataset.attrs.get(
        "_FillValue",
        None,
    )

    if fill_value is None:
        fill_value = dataset.attrs.get(
            "missing_value",
            None,
        )

    if fill_value is None:
        return None

    return float(
        np.asarray(fill_value).squeeze()
    )


# Number of time samples read per full-record test block.
block_samples = 100_000

# Small-read test positions throughout the file.
test_starts = [
    0,
    100_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    14_000_000,
]


with h5py.File(file_path, "r") as f:

    east_ds = f["East"]
    north_ds = f["North"]
    time_ds = f["time"]
    z_bin = f["Z_bin"][...].astype(np.float64)

    print("\n" + "=" * 70)
    print("ADCP-HR EAST/NORTH AVAILABILITY CHECK")

    print("\nEast shape:", east_ds.shape)
    print("North shape:", north_ds.shape)

    if east_ds.shape != north_ds.shape:
        raise ValueError(
            "East and North datasets have different shapes."
        )

    if east_ds.ndim != 3:
        raise ValueError(
            "Expected East/North dimensions to be "
            "(time, instrument, cell)."
        )

    number_of_samples = east_ds.shape[0]
    number_of_instruments = east_ds.shape[1]
    number_of_cells = east_ds.shape[2]

    if number_of_instruments != 1:
        print(
            "Warning: expected one instrument dimension, found",
            number_of_instruments,
        )

    if len(z_bin) != number_of_cells:
        raise ValueError(
            "Z_bin length does not match number of cells."
        )

    east_scale = scalar_attribute(
        east_ds,
        "scale_factor",
        1.0,
    )
    east_offset = scalar_attribute(
        east_ds,
        "add_offset",
        0.0,
    )

    north_scale = scalar_attribute(
        north_ds,
        "scale_factor",
        1.0,
    )
    north_offset = scalar_attribute(
        north_ds,
        "add_offset",
        0.0,
    )

    east_fill = get_fill_value(east_ds)
    north_fill = get_fill_value(north_ds)

    print("\nNumber of samples:", number_of_samples)
    print("Number of cells:", number_of_cells)
    print("Z_bin values (m):")
    print(z_bin)

    # --------------------------------------------------------
    # INSPECT TIMESTAMPS AND SAMPLING FREQUENCY
    # --------------------------------------------------------

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

    # Read a limited initial portion to estimate regular sampling.
    time_test_count = min(
        200_000,
        time_ds.shape[0],
    )

    raw_time_test = (
        time_ds[:time_test_count]
        .astype(np.float64)
    )

    epoch_seconds_test = (
        raw_time_test * time_scale
        + time_offset
    )

    timestamps_test = pd.to_datetime(
        epoch_seconds_test,
        unit="s",
        utc=True,
        errors="coerce",
    )

    dt_seconds = (
        pd.Series(timestamps_test)
        .diff()
        .dt.total_seconds()
    )

    positive_dt = dt_seconds[
        np.isfinite(dt_seconds)
        & (dt_seconds > 0)
    ]

    median_regular_dt = float(
        positive_dt.quantile(0.25)
    )

    estimated_fs = (
        1.0 / median_regular_dt
        if median_regular_dt > 0
        else np.nan
    )

    first_raw_time = float(time_ds[0])
    last_raw_time = float(
        time_ds[time_ds.shape[0] - 1]
    )

    first_time = pd.to_datetime(
        first_raw_time * time_scale
        + time_offset,
        unit="s",
        utc=True,
    )

    last_time = pd.to_datetime(
        last_raw_time * time_scale
        + time_offset,
        unit="s",
        utc=True,
    )

    print("\nTime information")
    print("First timestamp:", first_time)
    print("Last timestamp: ", last_time)
    print(
        "Estimated regular dt:",
        median_regular_dt,
        "s",
    )
    print(
        "Estimated sampling frequency:",
        estimated_fs,
        "Hz",
    )

    # --------------------------------------------------------
    # SMALL-READ TESTS THROUGHOUT THE FILE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SMALL-READ TESTS")

    small_test_records = []

    for cell_index in range(number_of_cells):

        print(
            f"\nCell {cell_index + 1}, "
            f"Z_bin={z_bin[cell_index]:.3f} m"
        )

        for start in test_starts:

            if start >= number_of_samples:
                continue

            stop = min(
                start + 1000,
                number_of_samples,
            )

            try:
                east_raw = east_ds[
                    start:stop,
                    0,
                    cell_index,
                ].astype(np.float64)

                east_read_success = True
                east_error = ""

            except Exception as exc:
                east_raw = np.full(
                    stop - start,
                    np.nan,
                )

                east_read_success = False
                east_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            try:
                north_raw = north_ds[
                    start:stop,
                    0,
                    cell_index,
                ].astype(np.float64)

                north_read_success = True
                north_error = ""

            except Exception as exc:
                north_raw = np.full(
                    stop - start,
                    np.nan,
                )

                north_read_success = False
                north_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            if east_fill is not None:
                east_raw[east_raw == east_fill] = np.nan

            if north_fill is not None:
                north_raw[north_raw == north_fill] = np.nan

            east_values = (
                east_raw * east_scale
                + east_offset
            )

            north_values = (
                north_raw * north_scale
                + north_offset
            )

            east_valid = np.isfinite(east_values)
            north_valid = np.isfinite(north_values)

            paired_valid = (
                east_valid
                & north_valid
            )

            east_valid_fraction = (
                east_valid.mean()
            )

            north_valid_fraction = (
                north_valid.mean()
            )

            paired_valid_fraction = (
                paired_valid.mean()
            )

            print(
                f"{start:>9}: "
                f"East read={east_read_success}, "
                f"North read={north_read_success}, "
                f"paired valid="
                f"{paired_valid_fraction:.3f}"
            )

            if not east_read_success:
                print(
                    "  East error:",
                    east_error,
                )

            if not north_read_success:
                print(
                    "  North error:",
                    north_error,
                )

            small_test_records.append(
                {
                    "cell_number":
                        cell_index + 1,
                    "z_bin_m":
                        z_bin[cell_index],
                    "start_index":
                        start,
                    "stop_index":
                        stop,
                    "east_read_success":
                        east_read_success,
                    "north_read_success":
                        north_read_success,
                    "east_valid_fraction":
                        east_valid_fraction,
                    "north_valid_fraction":
                        north_valid_fraction,
                    "paired_valid_fraction":
                        paired_valid_fraction,
                    "east_error":
                        east_error,
                    "north_error":
                        north_error,
                }
            )

    small_test_results = pd.DataFrame(
        small_test_records
    )

    # --------------------------------------------------------
    # FULL-RECORD AVAILABILITY BY CELL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL-RECORD AVAILABILITY")

    east_valid_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    north_valid_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    paired_valid_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    east_readable_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    north_readable_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    east_failed_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    north_failed_count = np.zeros(
        number_of_cells,
        dtype=np.int64,
    )

    east_minimum = np.full(
        number_of_cells,
        np.nan,
    )

    east_maximum = np.full(
        number_of_cells,
        np.nan,
    )

    north_minimum = np.full(
        number_of_cells,
        np.nan,
    )

    north_maximum = np.full(
        number_of_cells,
        np.nan,
    )

    failure_records = []

    for block_start in range(
        0,
        number_of_samples,
        block_samples,
    ):

        block_stop = min(
            block_start + block_samples,
            number_of_samples,
        )

        block_length = (
            block_stop - block_start
        )

        print(
            f"Reading indices "
            f"{block_start}:{block_stop}"
        )

        for cell_index in range(
            number_of_cells
        ):

            # Read east
            try:
                east_raw = east_ds[
                    block_start:block_stop,
                    0,
                    cell_index,
                ].astype(np.float64)

                east_readable_count[
                    cell_index
                ] += block_length

                if east_fill is not None:
                    east_raw[
                        east_raw == east_fill
                    ] = np.nan

                east_values = (
                    east_raw * east_scale
                    + east_offset
                )

            except Exception as exc:
                east_values = np.full(
                    block_length,
                    np.nan,
                )

                east_failed_count[
                    cell_index
                ] += block_length

                failure_records.append(
                    {
                        "variable": "East",
                        "cell_number":
                            cell_index + 1,
                        "z_bin_m":
                            z_bin[cell_index],
                        "start_index":
                            block_start,
                        "stop_index":
                            block_stop,
                        "error":
                            f"{type(exc).__name__}: {exc}",
                    }
                )

            # Read north
            try:
                north_raw = north_ds[
                    block_start:block_stop,
                    0,
                    cell_index,
                ].astype(np.float64)

                north_readable_count[
                    cell_index
                ] += block_length

                if north_fill is not None:
                    north_raw[
                        north_raw == north_fill
                    ] = np.nan

                north_values = (
                    north_raw * north_scale
                    + north_offset
                )

            except Exception as exc:
                north_values = np.full(
                    block_length,
                    np.nan,
                )

                north_failed_count[
                    cell_index
                ] += block_length

                failure_records.append(
                    {
                        "variable": "North",
                        "cell_number":
                            cell_index + 1,
                        "z_bin_m":
                            z_bin[cell_index],
                        "start_index":
                            block_start,
                        "stop_index":
                            block_stop,
                        "error":
                            f"{type(exc).__name__}: {exc}",
                    }
                )

            east_finite = np.isfinite(
                east_values
            )

            north_finite = np.isfinite(
                north_values
            )

            paired_finite = (
                east_finite
                & north_finite
            )

            east_valid_count[
                cell_index
            ] += east_finite.sum()

            north_valid_count[
                cell_index
            ] += north_finite.sum()

            paired_valid_count[
                cell_index
            ] += paired_finite.sum()

            if east_finite.any():
                block_min = np.nanmin(
                    east_values
                )
                block_max = np.nanmax(
                    east_values
                )

                if np.isnan(
                    east_minimum[cell_index]
                ):
                    east_minimum[
                        cell_index
                    ] = block_min
                else:
                    east_minimum[
                        cell_index
                    ] = min(
                        east_minimum[
                            cell_index
                        ],
                        block_min,
                    )

                if np.isnan(
                    east_maximum[cell_index]
                ):
                    east_maximum[
                        cell_index
                    ] = block_max
                else:
                    east_maximum[
                        cell_index
                    ] = max(
                        east_maximum[
                            cell_index
                        ],
                        block_max,
                    )

            if north_finite.any():
                block_min = np.nanmin(
                    north_values
                )
                block_max = np.nanmax(
                    north_values
                )

                if np.isnan(
                    north_minimum[cell_index]
                ):
                    north_minimum[
                        cell_index
                    ] = block_min
                else:
                    north_minimum[
                        cell_index
                    ] = min(
                        north_minimum[
                            cell_index
                        ],
                        block_min,
                    )

                if np.isnan(
                    north_maximum[cell_index]
                ):
                    north_maximum[
                        cell_index
                    ] = block_max
                else:
                    north_maximum[
                        cell_index
                    ] = max(
                        north_maximum[
                            cell_index
                        ],
                        block_max,
                    )

    availability_records = []

    for cell_index in range(
        number_of_cells
    ):

        availability_records.append(
            {
                "cell_number":
                    cell_index + 1,
                "z_bin_m":
                    z_bin[cell_index],
                "east_valid_fraction":
                    east_valid_count[
                        cell_index
                    ] / number_of_samples,
                "north_valid_fraction":
                    north_valid_count[
                        cell_index
                    ] / number_of_samples,
                "paired_valid_fraction":
                    paired_valid_count[
                        cell_index
                    ] / number_of_samples,
                "east_readable_fraction":
                    east_readable_count[
                        cell_index
                    ] / number_of_samples,
                "north_readable_fraction":
                    north_readable_count[
                        cell_index
                    ] / number_of_samples,
                "east_failed_read_fraction":
                    east_failed_count[
                        cell_index
                    ] / number_of_samples,
                "north_failed_read_fraction":
                    north_failed_count[
                        cell_index
                    ] / number_of_samples,
                "east_minimum_m_s":
                    east_minimum[
                        cell_index
                    ],
                "east_maximum_m_s":
                    east_maximum[
                        cell_index
                    ],
                "north_minimum_m_s":
                    north_minimum[
                        cell_index
                    ],
                "north_maximum_m_s":
                    north_maximum[
                        cell_index
                    ],
            }
        )

availability = pd.DataFrame(
    availability_records
)

failures = pd.DataFrame(
    failure_records
)

print("\nAvailability by cell:")
print(
    availability.to_string(
        index=False
    )
)

print("\nSummary:")
print(
    availability[
        [
            "east_valid_fraction",
            "north_valid_fraction",
            "paired_valid_fraction",
            "east_failed_read_fraction",
            "north_failed_read_fraction",
        ]
    ].describe()
)

if failures.empty:
    print("\nNo HDF5 read failures detected.")
else:
    print("\nRead failures by variable and cell:")
    print(
        failures.groupby(
            [
                "variable",
                "cell_number",
            ]
        ).size()
    )