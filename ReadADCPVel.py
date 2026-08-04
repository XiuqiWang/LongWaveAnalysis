# -*- coding: utf-8 -*-
"""
Read KG2 ADCP-HR NetCDF data, rotate East/North velocity in all cells
into cross-shore and alongshore components, and save to Parquet.

The ADCP-HR velocity variables have dimensions:
    (time, instrument, cell)

For this file there is one instrument and 13 cells.

Important:
    Z_bin is the elevation of each bin relative to the bottom of the frame.
    It is not necessarily the exact elevation above the seabed.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

file_path = Path(
    r"C:\dev\Python\LongWaveAnalysis\ADCP-HR"
    r"\adcp_hr_dvn_201804_F1P1.nc"
)

case_id = "DVN_F1_ADCP_HR"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)
output_folder.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    output_folder
    / f"{case_id}_rotated.parquet"
)

cell_metadata_file = (
    output_folder
    / f"{case_id}_cell_metadata.csv"
)

failure_file = (
    output_folder
    / f"{case_id}_read_failures.csv"
)

# Positive cross-shore direction: nearshore -> offshore.
nearshore_point = (52.23317, 4.3873215)  # DVN F3
offshore_point = (52.28087, 4.2432895)   # DVN F1

# Select one instrument when the middle dimension has length > 1.
instrument_index = 0

# Set either value to None to read the complete available record.
target_time_start = pd.Timestamp(
    "2018-04-04 13:00:00",
    tz="UTC",
)

target_time_stop = pd.Timestamp(
    "2018-05-18 07:00:00",
    tz="UTC",
)

# Number of time samples requested in one read operation.
block_samples = 100_000

# Save original ENU components as well as rotated components.
keep_east_north = False

# Optionally save vertical velocity.
include_up = False


# ============================================================
# HELPERS
# ============================================================

def scalar_attribute(dataset, name, default):
    """Return a scalar HDF5/NetCDF attribute."""
    value = dataset.attrs.get(
        name,
        default,
    )

    return float(
        np.asarray(value).squeeze()
    )


def dataset_fill_value(dataset):
    """Return _FillValue or missing_value, if one exists."""
    value = dataset.attrs.get(
        "_FillValue",
        None,
    )

    if value is None:
        value = dataset.attrs.get(
            "missing_value",
            None,
        )

    if value is None:
        return None

    return float(
        np.asarray(value).squeeze()
    )


def decode_single_string(value):
    """Decode one HDF5 byte/string value."""
    value = np.asarray(value).reshape(-1)[0]

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def find_time_index(
    time_ds,
    target_time,
):
    """
    Return the first index at or after target_time.

    The complete time coordinate is not loaded into memory.
    """
    target_time = pd.Timestamp(
        target_time
    )

    if target_time.tzinfo is None:
        target_time = target_time.tz_localize(
            "UTC"
        )
    else:
        target_time = target_time.tz_convert(
            "UTC"
        )

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
        middle = (
            low + high
        ) // 2

        middle_value = float(
            time_ds[middle]
        )

        if middle_value < target_raw:
            low = middle + 1
        else:
            high = middle

    return low


def read_1d_chunked(
    dataset,
    start,
    stop,
    variable_name,
    block_samples,
):
    """
    Read a one-dimensional dataset in blocks.

    Unreadable blocks remain NaN, while later blocks are still attempted.
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

    fill_value = dataset_fill_value(
        dataset
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

        destination_start = (
            block_start - start
        )

        destination_stop = (
            block_stop - start
        )

        try:
            raw = dataset[
                block_start:block_stop
            ].astype(np.float64)

            if fill_value is not None:
                raw[
                    raw == fill_value
                ] = np.nan

            output[
                destination_start:
                destination_stop
            ] = (
                raw * scale
                + offset
            )

        except (
            OSError,
            RuntimeError,
        ) as exc:
            failures.append(
                {
                    "variable":
                        variable_name,
                    "cell_index":
                        np.nan,
                    "cell_number":
                        np.nan,
                    "start_index":
                        block_start,
                    "stop_index":
                        block_stop,
                    "error":
                        str(exc),
                }
            )

    return output, failures


def read_time_instrument_cells_chunked(
    dataset,
    start,
    stop,
    instrument_index,
    variable_name,
    block_samples,
):
    """
    Read an ADCP-HR variable with dimensions (time, instrument, cell).

    Unreadable cell/block combinations remain NaN. Reading continues
    after failures.
    """
    if dataset.ndim != 3:
        raise ValueError(
            f"{variable_name} must have dimensions "
            "(time, instrument, cell); "
            f"found shape {dataset.shape}."
        )

    number_of_instruments = (
        dataset.shape[1]
    )

    number_of_cells = (
        dataset.shape[2]
    )

    if not (
        0
        <= instrument_index
        < number_of_instruments
    ):
        raise IndexError(
            f"instrument_index={instrument_index} "
            f"is invalid for {number_of_instruments} instruments."
        )

    output = np.full(
        (
            stop - start,
            number_of_cells,
        ),
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

    fill_value = dataset_fill_value(
        dataset
    )

    for cell_index in range(
        number_of_cells
    ):
        for block_start in range(
            start,
            stop,
            block_samples,
        ):
            block_stop = min(
                block_start + block_samples,
                stop,
            )

            destination_start = (
                block_start - start
            )

            destination_stop = (
                block_stop - start
            )

            try:
                raw = dataset[
                    block_start:block_stop,
                    instrument_index,
                    cell_index,
                ].astype(np.float64)

                if fill_value is not None:
                    raw[
                        raw == fill_value
                    ] = np.nan

                output[
                    destination_start:
                    destination_stop,
                    cell_index,
                ] = (
                    raw * scale
                    + offset
                )

            except (
                OSError,
                RuntimeError,
            ) as exc:
                failures.append(
                    {
                        "variable":
                            variable_name,
                        "cell_index":
                            cell_index,
                        "cell_number":
                            cell_index + 1,
                        "start_index":
                            block_start,
                        "stop_index":
                            block_stop,
                        "error":
                            str(exc),
                    }
                )

    return output, failures


def rotation_information(
    point_from,
    point_to,
):
    """
    Return cross-shore and alongshore unit vectors.

    point_from -> point_to defines positive cross-shore.
    Positive alongshore is 90 degrees counterclockwise from it.
    """
    lat1, lon1 = map(
        float,
        point_from,
    )

    lat2, lon2 = map(
        float,
        point_to,
    )

    earth_radius_m = 6_371_000.0

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

    mean_lat_rad = 0.5 * (
        lat1_rad + lat2_rad
    )

    delta_east_m = (
        earth_radius_m
        * (lon2_rad - lon1_rad)
        * np.cos(mean_lat_rad)
    )

    delta_north_m = (
        earth_radius_m
        * (lat2_rad - lat1_rad)
    )

    transect_length_m = np.hypot(
        delta_east_m,
        delta_north_m,
    )

    if transect_length_m == 0:
        raise ValueError(
            "Transect points must be different."
        )

    cross_east = (
        delta_east_m
        / transect_length_m
    )

    cross_north = (
        delta_north_m
        / transect_length_m
    )

    along_east = -cross_north
    along_north = cross_east

    bearing_deg = (
        np.rad2deg(
            np.arctan2(
                delta_east_m,
                delta_north_m,
            )
        )
        + 360.0
    ) % 360.0

    return {
        "point_from_lat":
            lat1,
        "point_from_lon":
            lon1,
        "point_to_lat":
            lat2,
        "point_to_lon":
            lon2,
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
        "positive_alongshore_definition":
            (
                "90 degrees counterclockwise "
                "from positive cross-shore"
            ),
    }


# ============================================================
# READ DATA
# ============================================================

all_failures = []

with h5py.File(
    file_path,
    "r",
) as f:
    required_variables = {
        "time",
        "cell",
        "Z_bin",
        "East",
        "North",
    }

    missing_variables = (
        required_variables
        - set(f.keys())
    )

    if missing_variables:
        raise KeyError(
            "Missing required variables: "
            f"{sorted(missing_variables)}"
        )

    time_ds = f["time"]
    east_ds = f["East"]
    north_ds = f["North"]

    if east_ds.shape != north_ds.shape:
        raise ValueError(
            "East and North have different shapes."
        )

    if east_ds.ndim != 3:
        raise ValueError(
            "East and North must have shape "
            "(time, instrument, cell)."
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

    cell_values = np.asarray(
        f["cell"][...]
    ).reshape(-1)

    z_bin_m = np.asarray(
        f["Z_bin"][...],
        dtype=np.float64,
    ).reshape(-1)

    number_of_cells = (
        east_ds.shape[2]
    )

    if len(z_bin_m) != number_of_cells:
        raise ValueError(
            "Z_bin length does not match "
            "the velocity cell count."
        )

    if len(cell_values) != number_of_cells:
        raise ValueError(
            "cell dimension length does not match "
            "the velocity cell count."
        )

    instrument_name = (
        decode_single_string(
            f["instrument"][...]
        )
        if "instrument" in f
        else f"instrument_{instrument_index}"
    )

    instrument_height_in_frame_m = (
        float(
            np.asarray(
                f["Z"][...]
            ).reshape(-1)[
                instrument_index
            ]
        )
        if "Z" in f
        else np.nan
    )

    epoch_seconds, failures = (
        read_1d_chunked(
            dataset=time_ds,
            start=start_index,
            stop=stop_index,
            variable_name="time",
            block_samples=block_samples,
        )
    )

    all_failures.extend(
        failures
    )

    east, failures = (
        read_time_instrument_cells_chunked(
            dataset=east_ds,
            start=start_index,
            stop=stop_index,
            instrument_index=instrument_index,
            variable_name="East",
            block_samples=block_samples,
        )
    )

    all_failures.extend(
        failures
    )

    north, failures = (
        read_time_instrument_cells_chunked(
            dataset=north_ds,
            start=start_index,
            stop=stop_index,
            instrument_index=instrument_index,
            variable_name="North",
            block_samples=block_samples,
        )
    )

    all_failures.extend(
        failures
    )

    up = None

    if include_up:
        if "Up" not in f:
            raise KeyError(
                "include_up=True, but Up is missing."
            )

        up, failures = (
            read_time_instrument_cells_chunked(
                dataset=f["Up"],
                start=start_index,
                stop=stop_index,
                instrument_index=instrument_index,
                variable_name="Up",
                block_samples=block_samples,
            )
        )

        all_failures.extend(
            failures
        )


# ============================================================
# ROTATE ALL CELLS
# ============================================================

rotation = rotation_information(
    nearshore_point,
    offshore_point,
)

cross_shore = (
    east
    * rotation[
        "cross_shore_unit_east"
    ]
    + north
    * rotation[
        "cross_shore_unit_north"
    ]
)

alongshore = (
    east
    * rotation[
        "alongshore_unit_east"
    ]
    + north
    * rotation[
        "alongshore_unit_north"
    ]
)

print("\nInstrument:", instrument_name)
print(
    "Instrument height in frame:",
    instrument_height_in_frame_m,
    "m",
)

print("\nCell identifiers:")
print(cell_values)

print("\nZ_bin relative to frame bottom (m):")
print(z_bin_m)

print("\nRotation information:")
for key, value in rotation.items():
    print(f"{key}: {value}")


# ============================================================
# DATA AVAILABILITY DIAGNOSTICS
# ============================================================

print("\nValid fractions before rotation:")

for cell_index in range(
    number_of_cells
):
    east_valid = np.isfinite(
        east[:, cell_index]
    ).mean()

    north_valid = np.isfinite(
        north[:, cell_index]
    ).mean()

    paired_valid = (
        np.isfinite(
            east[:, cell_index]
        )
        & np.isfinite(
            north[:, cell_index]
        )
    ).mean()

    print(
        f"Cell {cell_index + 1:02d}, "
        f"Z_bin={z_bin_m[cell_index]:.3f} m: "
        f"East={east_valid:.6f}, "
        f"North={north_valid:.6f}, "
        f"paired={paired_valid:.6f}"
    )


# ============================================================
# BUILD WIDE OUTPUT TABLE
# ============================================================

result = pd.DataFrame(
    {
        "time": pd.to_datetime(
            epoch_seconds,
            unit="s",
            utc=True,
            errors="coerce",
        )
    }
)

for cell_index in range(
    number_of_cells
):
    cell_number = (
        cell_index + 1
    )

    cell_id = int(
        cell_values[cell_index]
    )

    # Preserve the sign of negative Z_bin values in the column name.
    z_mm = int(
        round(
            z_bin_m[cell_index]
            * 1000.0
        )
    )

    if z_mm < 0:
        z_label = (
            f"zm{abs(z_mm):04d}mm"
        )
    else:
        z_label = (
            f"zp{z_mm:04d}mm"
        )

    suffix = (
        f"cell{cell_number:02d}"
        f"_id{cell_id:02d}"
        f"_{z_label}"
    )

    if keep_east_north:
        result[
            f"east_{suffix}_m_s"
        ] = east[
            :,
            cell_index,
        ]

        result[
            f"north_{suffix}_m_s"
        ] = north[
            :,
            cell_index,
        ]

    result[
        f"cross_shore_{suffix}_m_s"
    ] = cross_shore[
        :,
        cell_index,
    ]

    result[
        f"alongshore_{suffix}_m_s"
    ] = alongshore[
        :,
        cell_index,
    ]

    if include_up:
        result[
            f"up_{suffix}_m_s"
        ] = up[
            :,
            cell_index,
        ]


# Rows with unreadable time cannot be placed reliably.
number_of_missing_times = (
    result["time"].isna().sum()
)

if number_of_missing_times:
    print(
        "\nDropping",
        number_of_missing_times,
        "rows with unreadable time.",
    )

    result = (
        result.dropna(
            subset=["time"]
        )
        .reset_index(
            drop=True
        )
    )

if result.empty:
    raise RuntimeError(
        "No rows remain after reading time."
    )

print("\nExtracted ADCP-HR data")
print("Start:", result["time"].iloc[0])
print("End:  ", result["time"].iloc[-1])
print("Rows: ", len(result))

print("\nMissing values by column:")
print(
    result.isna().sum()
)


# ============================================================
# CELL METADATA
# ============================================================

cell_metadata = pd.DataFrame(
    {
        "case_id":
            case_id,
        "instrument_name":
            instrument_name,
        "instrument_index":
            instrument_index,
        "instrument_height_in_frame_m":
            instrument_height_in_frame_m,
        "cell_number":
            np.arange(
                1,
                number_of_cells + 1,
            ),
        "cell_id_from_file":
            cell_values.astype(int),
        "z_bin_relative_to_frame_bottom_m":
            z_bin_m,
    }
)

for key, value in rotation.items():
    cell_metadata[key] = value


# ============================================================
# SAVE OUTPUTS
# ============================================================

result.to_parquet(
    output_file,
    index=False,
)

cell_metadata.to_csv(
    cell_metadata_file,
    index=False,
)

if all_failures:
    pd.DataFrame(
        all_failures
    ).to_csv(
        failure_file,
        index=False,
    )

    print("\nRead failures saved to:")
    print(failure_file)

    print(
        "Number of failed cell/block reads:",
        len(all_failures),
    )
else:
    print(
        "\nNo HDF5 read failures detected."
    )

print("\nSaved rotated ADCP-HR data to:")
print(output_file)

print("\nSaved ADCP-HR cell metadata to:")
print(cell_metadata_file)