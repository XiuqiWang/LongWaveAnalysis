# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:36:47 2026

@author: WangX3

Reads adv netcdf files and finds the first active time window, and prints the block near the active window.
For Frame 1, there are two columns -> two elevations

"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

def _scalar_attribute(dataset, name, default):
    value = dataset.attrs.get(name, default)
    return float(np.asarray(value).squeeze())


def _read_velocity_chunked(
    dataset,
    start_index,
    stop_index,
    column,
    block_samples=100_000,
):
    """
    Read one velocity column in blocks.

    Unreadable blocks are filled with NaN.
    """
    number_of_samples = stop_index - start_index
    output = np.full(number_of_samples, np.nan, dtype=np.float64)
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

    fill_value = dataset.attrs.get("_FillValue", None)

    if fill_value is not None:
        fill_value = float(np.asarray(fill_value).squeeze())

    for block_start in range(
        start_index,
        stop_index,
        block_samples,
    ):
        block_stop = min(
            block_start + block_samples,
            stop_index,
        )

        destination_start = block_start - start_index
        destination_stop = block_stop - start_index

        try:
            raw = dataset[
                block_start:block_stop,
                column,
            ].astype(np.float64)

            if fill_value is not None:
                raw[raw == fill_value] = np.nan

            output[destination_start:destination_stop] = (
                raw * scale + offset
            )

        except OSError as exc:
            failures.append(
                {
                    "start_index": block_start,
                    "stop_index": block_stop,
                    "error": str(exc),
                }
            )

    return output, failures


def read_adv_block_chunked(
    file_path,
    adv_column,
    instrument,
    z_velocity_m,
    start_index,
    stop_index,
    block_samples=100_000,
    include_up=False,
):
    """
    Read ADV01 only from a KG2 ADV file.

    ADV01:
        NetCDF column = 0
        velocity measurement height = 0.488 m above frame/seabed

    Unreadable compressed chunks remain NaN in the returned DataFrame.

    Returns
    -------
    result : pandas.DataFrame
        Time and ADV01 physical velocities in m/s.
    failures : list[dict]
        Unreadable ADV01 variable/index ranges.
    """
    file_path = Path(file_path)

    z_velocity_m = 0.488
    height_cm = round(z_velocity_m * 100)

    variable_components = [
        ("East_despiked", "east"),
        ("North_despiked", "north"),
    ]

    if include_up:
        variable_components.append(
            ("Up_despiked", "up")
        )

    all_failures = []

    with h5py.File(file_path, "r") as f:
        time_ds = f["time"]
        number_of_samples = time_ds.shape[0]

        start_index = max(0, int(start_index))
        stop_index = min(int(stop_index), number_of_samples)

        if stop_index <= start_index:
            raise ValueError(
                "stop_index must be greater than start_index."
            )

        number_to_read = stop_index - start_index

        # Start filled with NaN so failed time chunks remain identifiable.
        raw_time = np.full(
            number_to_read,
            np.nan,
            dtype=np.float64,
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

            destination_start = block_start - start_index
            destination_stop = block_stop - start_index

            try:
                raw_time[destination_start:destination_stop] = (
                    time_ds[block_start:block_stop]
                    .astype(np.float64)
                )
            except OSError as exc:
                all_failures.append(
                    {
                        "variable": "time",
                        "instrument": instrument,
                        "column": adv_column,
                        "start_index": block_start,
                        "stop_index": block_stop,
                        "error": str(exc),
                    }
                )

        time_scale = _scalar_attribute(
            time_ds,
            "scale_factor",
            1.0,
        )
        time_offset = _scalar_attribute(
            time_ds,
            "add_offset",
            0.0,
        )

        epoch_seconds = (
            raw_time * time_scale + time_offset
        )

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

        for variable_name, component in variable_components:
            dataset = f[variable_name]

            values, failures = _read_velocity_chunked(
                dataset=dataset,
                start_index=start_index,
                stop_index=stop_index,
                column=adv_column,
                block_samples=block_samples,
            )

            output_name = (
                f"{component}_{instrument}_{height_cm}cm"
            )

            result[output_name] = values

            for failure in failures:
                failure.update(
                    {
                        "variable": variable_name,
                        "instrument": instrument,
                        "column": adv_column,
                    }
                )
                all_failures.append(failure)

    return result, all_failures

def find_first_active_window(
    file_path,
    column,
    fs=16.0,
    window_seconds=60,
    std_threshold=0.002,
):
    """
    Find the first sustained active window for one ADV instrument.

    Activity is detected from the combined variability of the despiked
    East and North velocity components.

    Parameters
    ----------
    file_path : str or Path
        KG2 ADV NetCDF file.
    column : int
        0 for ADV01, 1 for ADV02.
    fs : float
        Sampling frequency in Hz.
    window_seconds : float
        Detection-window length in seconds.
    std_threshold : float
        Minimum combined velocity standard deviation in m/s.

    Returns
    -------
    index : int or None
        First sample index of the detected active window.
    timestamp : pandas.Timestamp or None
        UTC timestamp at that index.
    diagnostics : dict or None
        Standard deviations of each velocity component.
    """
    window_size = int(round(fs * window_seconds))
    block_size = int(round(fs * 3600))

    with h5py.File(file_path, "r") as f:
        time_ds = f["time"]

        component_names = [
            "East_despiked",
            "North_despiked",
        ]

        datasets = {
            name: f[name]
            for name in component_names
        }

        scales = {
            name: float(
                np.asarray(
                    datasets[name].attrs.get("scale_factor", 1.0)
                ).squeeze()
            )
            for name in component_names
        }

        offsets = {
            name: float(
                np.asarray(
                    datasets[name].attrs.get("add_offset", 0.0)
                ).squeeze()
            )
            for name in component_names
        }

        n_samples = time_ds.shape[0]

        for block_start in range(0, n_samples, block_size):
            block_stop = min(block_start + block_size, n_samples)

            block = {}

            for name in component_names:
                block[name] = (
                    datasets[name][block_start:block_stop, column]
                    .astype(np.float64)
                    * scales[name]
                    + offsets[name]
                )

            for local_start in range(
                0,
                block_stop - block_start - window_size + 1,
                window_size,
            ):
                local_stop = local_start + window_size

                east = block["East_despiked"][local_start:local_stop]
                north = block["North_despiked"][local_start:local_stop]

                valid = (
                    np.isfinite(east)
                    & np.isfinite(north)
                )

                if valid.mean() <= 0.95:
                    continue

                east_std = np.nanstd(east)
                north_std = np.nanstd(north)

                combined_std = np.sqrt(
                    east_std**2
                    + north_std**2
                )

                if combined_std > std_threshold:
                    index = block_start + local_start

                    raw_time = float(time_ds[index])
                    time_scale = float(
                        np.asarray(
                            time_ds.attrs["scale_factor"]
                        ).squeeze()
                    )
                    time_offset = float(
                        np.asarray(
                            time_ds.attrs["add_offset"]
                        ).squeeze()
                    )

                    timestamp = pd.to_datetime(
                        raw_time * time_scale + time_offset,
                        unit="s",
                        utc=True,
                    )

                    diagnostics = {
                        "east_std_m_s": east_std,
                        "north_std_m_s": north_std,
                        "combined_std_m_s": combined_std,
                    }

                    return index, timestamp, diagnostics

    return None, None, None

def find_time_index(
time_ds,
target_time,
):
    """
    Find the first NetCDF index at or after target_time.

    The time coordinate must be monotonically increasing.
    The complete time array is not loaded into memory.
    """
    target_time = pd.Timestamp(target_time)

    if target_time.tzinfo is None:
        target_time = target_time.tz_localize("UTC")
    else:
        target_time = target_time.tz_convert("UTC")

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
        middle_value = float(time_ds[middle])

        if middle_value < target_raw:
            low = middle + 1
        else:
            high = middle

    return low

def add_failure_times(failures, file_path):
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

            t0 = float(time_ds[i0]) * scale + offset
            t1 = float(time_ds[i1]) * scale + offset

            failure["start_time_utc"] = pd.to_datetime(
                t0,
                unit="s",
                utc=True,
            )
            failure["end_time_utc"] = pd.to_datetime(
                t1,
                unit="s",
                utc=True,
            )

    return failures

def rotate_enu_to_cross_along(
    east,
    north,
    point_from,
    point_to,
):
    """
    Rotate eastward/northward velocities into cross-shore and alongshore
    components using two user-defined geographic points.

    Parameters
    ----------
    east : array-like
        Eastward velocity in m/s.
    north : array-like
        Northward velocity in m/s.
    point_from : tuple(float, float)
        Starting point of the positive cross-shore transect as
        (latitude_deg, longitude_deg).
    point_to : tuple(float, float)
        Ending point of the positive cross-shore transect as
        (latitude_deg, longitude_deg).

        The direction point_from -> point_to defines positive cross-shore.
        For positive-offshore velocity, use:
            point_from = nearshore point
            point_to   = offshore point

    Returns
    -------
    cross_shore : numpy.ndarray
        Velocity parallel to the transect. Positive from point_from
        toward point_to.
    alongshore : numpy.ndarray
        Velocity 90 degrees counterclockwise from the positive
        cross-shore direction.
    metadata : dict
        Transect bearing and unit-vector information.

    Notes
    -----
    Latitude and longitude are converted locally into east/north distances.
    This approximation is accurate for a short coastal transect.
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)

    if east.shape != north.shape:
        raise ValueError("east and north must have the same shape.")

    lat1, lon1 = map(float, point_from)
    lat2, lon2 = map(float, point_to)

    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    # Local equirectangular conversion from geographic differences
    # to eastward and northward distances.
    earth_radius_m = 6_371_000.0

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)

    mean_lat_rad = 0.5 * (lat1_rad + lat2_rad)

    delta_east_m = (
        earth_radius_m
        * (lon2_rad - lon1_rad)
        * np.cos(mean_lat_rad)
    )
    delta_north_m = earth_radius_m * (lat2_rad - lat1_rad)

    transect_length_m = np.hypot(delta_east_m, delta_north_m)

    if transect_length_m == 0:
        raise ValueError("point_from and point_to must be different.")

    # Unit vector in the positive cross-shore direction.
    cross_east = delta_east_m / transect_length_m
    cross_north = delta_north_m / transect_length_m

    # Alongshore unit vector: 90 degrees counterclockwise from cross-shore.
    along_east = -cross_north
    along_north = cross_east

    # Vector projections.
    cross_shore = east * cross_east + north * cross_north
    alongshore = east * along_east + north * along_north

    # Bearing clockwise from geographic north.
    bearing_deg = (
        np.rad2deg(np.arctan2(delta_east_m, delta_north_m))
        + 360.0
    ) % 360.0

    metadata = {
        "point_from_lat_lon": (lat1, lon1),
        "point_to_lat_lon": (lat2, lon2),
        "positive_cross_shore_bearing_deg": bearing_deg,
        "transect_length_m": transect_length_m,
        "cross_shore_unit_east": cross_east,
        "cross_shore_unit_north": cross_north,
        "alongshore_unit_east": along_east,
        "alongshore_unit_north": along_north,
        "positive_alongshore_definition":
            "90 degrees counterclockwise from positive cross-shore",
    }

    return cross_shore, alongshore, metadata


file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADV"
    r"\adv_dvn_201804_F1.nc"
)

adv_column = 0
instrument_name = "ADV01"

#find the first active velocity time
index, timestamp, diagnostics = find_first_active_window(
    file_path=file_path,
    column=adv_column, 
    fs=16.0,
    window_seconds=60,
    std_threshold=0.001,
)

# print("\n", "ADV01")
# print("Index:", index)
# print("UTC time:", timestamp)
# print("Diagnostics:", diagnostics)

target_time_stop = pd.Timestamp("2018-05-15 13:30:00", tz="UTC")

with h5py.File(file_path, "r") as f:
    stop = find_time_index(
        time_ds=f["time"],
        target_time=target_time_stop,
    )

start = index

df, failures = read_adv_block_chunked(
    file_path=file_path,
    adv_column=0,
    instrument="ADV01",
    z_velocity_m=0.488,
    start_index=start,
    stop_index=stop,
    block_samples=100_000,
    include_up=False,
)

print("\nExtracted ADV01 data")
print("Start:", df["time"].iloc[0])
print("End:", df["time"].iloc[-1])
print("Rows:", len(df))

print("\nMissing values before rotation:")
print(df.isna().sum())

# print("\nUnreadable blocks:")
# for failure in failures:
#     print(failure)

# Where the failed data is
# failures = add_failure_times(
#     failures,
#     file_path,
# )

# for failure in failures:
#     print(
#         failure["variable"],
#         failure["instrument"],
#         failure["start_time_utc"],
#         "to",
#         failure["end_time_utc"],
#     )

# Coordinates of the selected transect points
nearshore_point = (52.2300, 4.3900) #DVN1
offshore_point = (52.2800, 4.2400) #DVN3

# Rotate for ADV01 (higher one) velocity
df["cross_shore"], df["alongshore"], rotation_info = (
    rotate_enu_to_cross_along(
        east=df["east_ADV01_49cm"],
        north=df["north_ADV01_49cm"],
        point_from=nearshore_point,
        point_to=offshore_point,
    )
)

# print(rotation_info)
# print(
#     df[
#         [
#             "time",
#             "east_ADV01_49cm",
#             "north_ADV01_49cm",
#             "cross_shore",
#             "alongshore",
#         ]
#     ].head()
# )    

# Retain only variables relevant to ADV01 processing.
df = df[
    [
        "time",
        "east_ADV01_49cm",
        "north_ADV01_49cm",
        "cross_shore",
        "alongshore",
    ]
].copy()


# Save processed ADV data.
output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
)
output_folder.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    output_folder
    / "ADV_F1_ADV01_rotated.parquet"
)

df.to_parquet(
    output_file,
    index=False,
)

print("\nSaved processed ADV01 data to:")
print(output_file)