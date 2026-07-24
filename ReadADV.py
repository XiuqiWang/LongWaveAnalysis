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


def read_adv_block(
    file_path,
    start_index,
    stop_index,
):
    """
    Read one index range from a KG2 ADV file.

    Returns physical velocity in m/s and UTC timestamps.
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
        # Decode time
        time_ds = f["time"]
        raw_time = time_ds[start_index:stop_index].astype(np.float64)

        time_scale = float(
            np.asarray(time_ds.attrs["scale_factor"]).squeeze()
        )
        time_offset = float(
            np.asarray(time_ds.attrs["add_offset"]).squeeze()
        )

        epoch_seconds = raw_time * time_scale + time_offset
        time_utc = pd.to_datetime(epoch_seconds, unit="s", utc=True)

        result = pd.DataFrame({"time": time_utc})

        # Column mapping derived from instrument and Z_vel
        columns = {
            0: ("ADV01", 0.488),
            1: ("ADV02", 0.189),
        }

        for variable_name, component in [
            ("East_despiked", "east"),
            ("North_despiked", "north"),
            ("Up_despiked", "up"),
        ]:
            ds = f[variable_name]

            scale = float(
                np.asarray(ds.attrs.get("scale_factor", 1.0)).squeeze()
            )
            offset = float(
                np.asarray(ds.attrs.get("add_offset", 0.0)).squeeze()
            )

            for column, (instrument, z_vel) in columns.items():
                raw = ds[start_index:stop_index, column].astype(
                    np.float64
                )
                physical = raw * scale + offset

                height_cm = round(z_vel * 100)
                result[
                    f"{component}_{instrument}_{height_cm}cm"
                ] = physical

    return result

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
                        "up_std_m_s": up_std,
                        "combined_std_m_s": combined_std,
                    }

                    return index, timestamp, diagnostics

    return None, None, None

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADV"
    r"\adv_dvn_201804_F1.nc"
)

#find the first active velocity time
for column, name in [(0, "ADV01"), (1, "ADV02")]:
    index, timestamp, diagnostics = find_first_active_window(
        file_path=file_path,
        column=column,
        fs=16.0,
        window_seconds=60,
        std_threshold=0.002,
    )
    
    print("\n", name)
    print("Index:", index)
    print("UTC time:", timestamp)
    print("Diagnostics:", diagnostics)

    #read the block from this time
    fs = 16.0 #16 samples per second (16 Hz)
    start = index - int(1800 * fs) # half an hour before the detected time
    stop = index + int(1800 * fs) # half an hour after the detected time
    
    df = read_adv_block(
        file_path,
        start_index=start,
        stop_index=stop,
    )
    
    print(df.head())
    print(df.columns)