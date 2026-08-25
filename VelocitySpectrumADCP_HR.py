# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:48:23 2026

@author: WangX3
Compute cross-shore and alongshore ADCP velocity autospectra for all cells.

Workflow
--------
1. Read the rotated ADCP Parquet file.
2. Detect the sampling frequency from the timestamps.
3. Detect natural continuous recording segments from timestamp gaps.
4. Detect the nominal burst length and split long segments into burst blocks.
5. Process every ADCP cell independently within each burst.
6. Quadratically detrend velocity.
7. Calculate Welch autospectra and IG/sea-swell RMS velocities.
8. Save burst statistics and spectra.
9. Plot:
   - time-varying IG RMS for all cells;
   - the most energetic IG burst for all cells;
   - median autospectra for all cells;
   - mean autospectra for all cells.

The script expects column names produced by read_rotate_adcp_8cells.py, e.g.
    cross_shore_cell01_z123cm_m_s
    alongshore_cell01_z123cm_m_s
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import welch, csd

# ============================================================
# USER SETTINGS
# ============================================================

input_file = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
    r"\DVN_F1_ADCP_HR_rotated.parquet"
)

case_id = "DVN_F1_ADCP-HR"
case_label = "DVN F1 ADCP-HR"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)
output_folder.mkdir(parents=True, exist_ok=True)

# Only these cells are processed spectrally.
# All available cells are still used for the background current.
selected_cell_numbers = [1, 5, 9, 13]

# Natural-segment detection
gap_threshold_seconds = 2.0

# Welch settings
welch_segment_seconds = 512.0
overlap_fraction = 0.5

# Frequency bands
ig_low_hz = 0.005
ig_high_hz = 0.05

ss_low_hz = 0.05
ss_high_hz = 1.0

# QC
minimum_valid_fraction = 0.98
minimum_block_fraction = 0.95
minimum_velocity_std_m_s = 1e-4

# Detrending
detrend_order = 2

# Gap handling
maximum_interpolation_gap_seconds = 2.0
plot_gap_seconds = 3600.0

# ============================================================
# FUNCTIONS
# ============================================================

def identify_velocity_columns(columns):
    """Identify paired cross-shore and alongshore ADCP velocity columns."""

    pattern = re.compile(
        r"^(cross_shore|alongshore)_"
        r"cell(?P<cell>\d+)_"
        r"id(?P<cell_id>\d+)_"
        r"z(?P<sign>[pm])(?P<height_mm>\d+)mm_m_s$"
    )

    found = {}

    for column in columns:
        match = pattern.match(column)
        if match is None:
            continue

        component = match.group(1)
        cell_number = int(match.group("cell"))
        cell_id = int(match.group("cell_id"))
        height_mm = int(match.group("height_mm"))

        if match.group("sign") == "m":
            height_mm = -height_mm

        key = (cell_number, cell_id, height_mm)

        found.setdefault(
            key,
            {
                "cell_number": cell_number,
                "cell_id": cell_id,
                "z_bin_m": height_mm / 1000.0,
            },
        )

        found[key][component] = column

    cells = [
        item
        for item in found.values()
        if "cross_shore" in item
        and "alongshore" in item
    ]

    if not cells:
        raise KeyError(
            "No paired ADCP-HR cross-shore/alongshore columns found."
        )

    return sorted(
        cells,
        key=lambda cell: cell["cell_number"],
    )


def detect_sampling_frequency(time):
    """Estimate sampling frequency from positive timestamp intervals."""

    dt = (
        pd.Series(time)
        .diff()
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )

    positive = dt[
        np.isfinite(dt)
        & (dt > 0)
    ]

    if positive.size == 0:
        raise ValueError(
            "Could not estimate the sampling interval."
        )

    cutoff = np.nanpercentile(
        positive,
        25,
    )

    regular = positive[
        positive <= cutoff * 1.5
    ]

    if regular.size == 0:
        regular = positive

    median_dt = float(
        np.nanmedian(regular)
    )

    if median_dt <= 0:
        raise ValueError(
            "Detected a nonpositive sampling interval."
        )

    return 1.0 / median_dt, median_dt


def identify_continuous_segments(
    df,
    gap_threshold_seconds,
):
    """Assign a new segment ID at large/nonpositive time steps."""

    output = df.copy()

    dt_seconds = (
        output["time"]
        .diff()
        .dt.total_seconds()
    )

    new_segment = (
        dt_seconds.isna()
        | (dt_seconds > gap_threshold_seconds)
        | (dt_seconds <= 0)
    )

    output["segment_id"] = (
        new_segment.cumsum()
    )

    return output


def split_segment_into_analysis_blocks(
    segment,
    nominal_block_samples,
    minimum_block_samples,
):
    """Split a continuous segment into nominal burst-length blocks."""

    blocks = []
    start = 0
    subblock_number = 0
    n = len(segment)

    while n - start >= nominal_block_samples:
        stop = start + nominal_block_samples

        block = segment.iloc[
            start:stop
        ].copy()

        block[
            "subblock_number"
        ] = subblock_number

        blocks.append(block)

        start = stop
        subblock_number += 1

    remainder = segment.iloc[
        start:
    ].copy()

    if len(remainder) >= minimum_block_samples:
        remainder[
            "subblock_number"
        ] = subblock_number

        blocks.append(remainder)

    return blocks


def polynomial_detrend(
    values,
    order=2,
):
    """Remove polynomial background from a finite 1D array."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Input must be one-dimensional."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains non-finite values."
        )

    x = np.linspace(
        -1.0,
        1.0,
        len(values),
    )

    coefficients = np.polyfit(
        x,
        values,
        deg=order,
    )

    trend = np.polyval(
        coefficients,
        x,
    )

    return values - trend, trend


def interpolate_short_gaps(
    series,
    limit_samples,
):
    """Interpolate only short internal gaps."""

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .interpolate(
            method="linear",
            limit=limit_samples,
            limit_area="inside",
        )
    )


def calculate_depth_averaged_background_current(
    block,
    cells,
    maximum_interpolation_gap_samples,
    minimum_valid_fraction,
):
    """
    Calculate the full-profile burst-mean background-current vector.

    A burst mean is calculated for every valid ADCP cell first,
    then those cell-mean vectors are averaged vertically.
    """

    cell_mean_cross = []
    cell_mean_along = []

    for cell in cells:
        cross_column = cell["cross_shore"]
        along_column = cell["alongshore"]

        finite_pair = (
            block[
                [
                    cross_column,
                    along_column,
                ]
            ]
            .notna()
            .all(axis=1)
        )

        if finite_pair.mean() < minimum_valid_fraction:
            continue

        cross = interpolate_short_gaps(
            block[cross_column],
            maximum_interpolation_gap_samples,
        )

        along = interpolate_short_gaps(
            block[along_column],
            maximum_interpolation_gap_samples,
        )

        if not (
            cross.notna()
            & along.notna()
        ).all():
            continue

        cell_mean_cross.append(
            float(cross.mean())
        )

        cell_mean_along.append(
            float(along.mean())
        )

    if len(cell_mean_cross) == 0:
        return {
            "depth_avg_cross_current_m_s": np.nan,
            "depth_avg_along_current_m_s": np.nan,
            "depth_avg_current_speed_m_s": np.nan,
            "depth_avg_current_direction_deg": np.nan,
            "current_profile_cell_count": 0,
        }

    mean_cross = float(
        np.mean(cell_mean_cross)
    )

    mean_along = float(
        np.mean(cell_mean_along)
    )

    current_speed = float(
        np.hypot(
            mean_cross,
            mean_along,
        )
    )

    current_direction = float(
        np.degrees(
            np.arctan2(
                mean_along,
                mean_cross,
            )
        )
    )

    return {
        "depth_avg_cross_current_m_s":
            mean_cross,

        "depth_avg_along_current_m_s":
            mean_along,

        "depth_avg_current_speed_m_s":
            current_speed,

        "depth_avg_current_direction_deg":
            current_direction,

        "current_profile_cell_count":
            len(cell_mean_cross),
    }


def calculate_autospectra(
    cross_shore,
    alongshore,
    fs,
    segment_seconds,
    overlap_fraction,
):
    """Calculate Welch autospectra and U-V cross-spectrum."""

    cross = np.asarray(cross_shore, dtype=np.float64)
    along = np.asarray(alongshore, dtype=np.float64)

    if cross.shape != along.shape:
        raise ValueError("Cross-shore and alongshore arrays differ in shape.")

    if not np.isfinite(cross).all() or not np.isfinite(along).all():
        raise ValueError("Velocity arrays contain missing values.")

    nperseg = min(int(round(segment_seconds * fs)), len(cross))
    if nperseg < 16:
        raise ValueError("Analysis block too short for Welch analysis.")

    noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)

    frequency, cross_psd = welch(
        cross, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, detrend=False, scaling="density",
        return_onesided=True,
    )

    frequency_along, along_psd = welch(
        along, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, detrend=False, scaling="density",
        return_onesided=True,
    )

    frequency_uv, uv_csd = csd(
        cross, along, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, detrend=False, scaling="density",
        return_onesided=True,
    )

    if (
        not np.allclose(frequency, frequency_along)
        or not np.allclose(frequency, frequency_uv)
    ):
        raise RuntimeError("Spectral frequency grids differ.")

    return frequency, cross_psd, along_psd, uv_csd


def integrate_spectral_band(
    frequency,
    psd,
    lower_hz,
    upper_hz,
):
    """Integrate PSD over [lower_hz, upper_hz)."""

    frequency = np.asarray(
        frequency,
        dtype=float,
    )

    psd = np.asarray(
        psd,
        dtype=float,
    )

    selected = (
        np.isfinite(frequency)
        & np.isfinite(psd)
        & (frequency >= lower_hz)
        & (frequency < upper_hz)
    )

    if selected.sum() < 2:
        return np.nan

    return np.trapezoid(
        psd[selected],
        frequency[selected],
    )


# ============================================================
# DISCOVER ALL CELLS
# ============================================================

print("Reading schema:")
print(input_file)

parquet_file = pq.ParquetFile(
    input_file
)

available_columns = (
    parquet_file
    .schema_arrow
    .names
)

if "time" not in available_columns:
    raise KeyError(
        "Parquet file does not contain a time column."
    )

all_cells = identify_velocity_columns(
    available_columns
)

for cell in all_cells:
    cell[
        "height_relative_to_frame_bottom_m"
    ] = cell["z_bin_m"]


# ============================================================
# SELECT CELLS FOR SPECTRAL ANALYSIS
# ============================================================

available_cell_numbers = {
    cell["cell_number"]
    for cell in all_cells
}

missing_cells = [
    cell_number
    for cell_number
    in selected_cell_numbers
    if cell_number
    not in available_cell_numbers
]

if missing_cells:
    raise ValueError(
        f"Selected cells not found: {missing_cells}. "
        f"Available cells: "
        f"{sorted(available_cell_numbers)}"
    )

spectral_cells = [
    cell
    for cell in all_cells
    if cell["cell_number"]
    in selected_cell_numbers
]

spectral_cells = sorted(
    spectral_cells,
    key=lambda cell:
        selected_cell_numbers.index(
            cell["cell_number"]
        ),
)

print("\nAll cells used for background current:")

for cell in all_cells:
    print(
        f"Cell {cell['cell_number']}: "
        f"z={cell['height_relative_to_frame_bottom_m']:.3f} m"
    )

print("\nCells used for spectral analysis:")

for cell in spectral_cells:
    print(
        f"Cell {cell['cell_number']}: "
        f"z={cell['height_relative_to_frame_bottom_m']:.3f} m"
    )


# ============================================================
# LOAD TIME + ALL VELOCITY COLUMNS
# ============================================================
#
# All 13 cells are needed for the full-profile current.
# Only four cells will later undergo Welch spectral analysis.
# ============================================================

required_columns = ["time"]

for cell in all_cells:
    required_columns.extend(
        [
            cell["cross_shore"],
            cell["alongshore"],
        ]
    )

df = pd.read_parquet(
    input_file,
    columns=required_columns,
)

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
    errors="coerce",
)

df = (
    df.dropna(
        subset=["time"]
    )
    .sort_values("time")
    .drop_duplicates(
        subset=["time"],
        keep="first",
    )
    .reset_index(drop=True)
)

for column in required_columns[1:]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# SAMPLING FREQUENCY
# ============================================================

fs, regular_dt_seconds = (
    detect_sampling_frequency(
        df["time"]
    )
)

print("\nDetected sampling:")
print("dt =", regular_dt_seconds, "s")
print("fs =", fs, "Hz")

print("\nRecord:")
print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows: ", len(df))


# ============================================================
# IDENTIFY NATURAL SEGMENTS
# ============================================================

df = identify_continuous_segments(
    df,
    gap_threshold_seconds,
)

segment_summary = (
    df.groupby("segment_id")
    .agg(
        start_time=("time", "first"),
        end_time=("time", "last"),
        sample_count=("time", "size"),
    )
    .reset_index()
)

segment_summary[
    "duration_minutes"
] = (
    segment_summary[
        "sample_count"
    ]
    / fs
    / 60.0
)

segment_length_counts = (
    segment_summary[
        "sample_count"
    ]
    .value_counts()
)

nominal_burst_samples = int(
    segment_length_counts.index[0]
)

minimum_block_samples = int(
    round(
        nominal_burst_samples
        * minimum_block_fraction
    )
)

print(
    "\nNominal burst:",
    nominal_burst_samples,
    "samples =",
    nominal_burst_samples
    / fs
    / 60.0,
    "minutes",
)

print(
    "Minimum accepted block:",
    minimum_block_samples,
    "samples",
)


# ============================================================
# CREATE ANALYSIS BLOCKS
# ============================================================

analysis_blocks = []

for segment_id, segment in df.groupby(
    "segment_id",
    sort=True,
):

    blocks = (
        split_segment_into_analysis_blocks(
            segment=
                segment.reset_index(
                    drop=True
                ),
            nominal_block_samples=
                nominal_burst_samples,
            minimum_block_samples=
                minimum_block_samples,
        )
    )

    for block in blocks:
        block[
            "source_segment_id"
        ] = segment_id

        analysis_blocks.append(
            block
        )

print(
    "\nAnalysis blocks:",
    len(analysis_blocks),
)


# ============================================================
# COMPUTE CURRENT + SELECTED-CELL SPECTRA
# ============================================================
statistics_records = []
spectral_records = []
directional_velocity_records = []

maximum_interpolation_gap_samples = max(
    1,
    int(
        round(
            maximum_interpolation_gap_seconds
            * fs
        )
    ),
)

nan_results = {
    "cell_mean_cross_current_m_s":
        np.nan,

    "cell_mean_along_current_m_s":
        np.nan,

    "cross_background_change_m_s":
        np.nan,

    "along_background_change_m_s":
        np.nan,

    "cross_ig_variance_m2_s2":
        np.nan,

    "along_ig_variance_m2_s2":
        np.nan,

    "total_horizontal_ig_variance_m2_s2":
        np.nan,

    "cross_ss_variance_m2_s2":
        np.nan,

    "along_ss_variance_m2_s2":
        np.nan,

    "total_horizontal_ss_variance_m2_s2":
        np.nan,
    "ig_uv_covariance_m2_s2": np.nan,
    "ss_uv_covariance_m2_s2": np.nan,
    "cross_ig_rms_m_s":
        np.nan,

    "along_ig_rms_m_s":
        np.nan,

    "horizontal_ig_rms_m_s":
        np.nan,

    "cross_ss_rms_m_s":
        np.nan,

    "along_ss_rms_m_s":
        np.nan,

    "horizontal_ss_rms_m_s":
        np.nan,
}


for analysis_block_number, block in enumerate(
    analysis_blocks
):

    start_time = block[
        "time"
    ].iloc[0]

    end_time = block[
        "time"
    ].iloc[-1]

    mid_time = (
        start_time
        + (
            end_time
            - start_time
        ) / 2
    )

    source_segment_id = int(
        block[
            "source_segment_id"
        ].iloc[0]
    )

    subblock_number = int(
        block[
            "subblock_number"
        ].iloc[0]
    )

    sample_count = len(block)

    duration_minutes = (
        sample_count
        / fs
        / 60.0
    )


    # ========================================================
    # FULL-PROFILE BACKGROUND CURRENT
    # ALL 13 CELLS
    # ========================================================

    background_current = (
        calculate_depth_averaged_background_current(
            block=block,
            cells=all_cells,
            maximum_interpolation_gap_samples=
                maximum_interpolation_gap_samples,
            minimum_valid_fraction=
                minimum_valid_fraction,
        )
    )


    # ========================================================
    # SPECTRAL ANALYSIS
    # ONLY CELLS 1, 5, 9, 13
    # ========================================================

    for cell in spectral_cells:

        cell_number = (
            cell["cell_number"]
        )

        height_m = cell[
            "height_relative_to_frame_bottom_m"
        ]

        cross_column = (
            cell["cross_shore"]
        )

        along_column = (
            cell["alongshore"]
        )

        finite_pair = (
            block[
                [
                    cross_column,
                    along_column,
                ]
            ]
            .notna()
            .all(axis=1)
        )

        valid_before = (
            finite_pair.mean()
        )

        base_record = {
            "analysis_block_number":
                analysis_block_number,

            "source_segment_id":
                source_segment_id,

            "subblock_number":
                subblock_number,

            "cell_number":
                cell_number,

            "height_relative_to_frame_bottom_m":
                height_m,

            "start_time":
                start_time,

            "end_time":
                end_time,

            "mid_time":
                mid_time,

            "sample_count":
                sample_count,

            "duration_minutes":
                duration_minutes,

            "valid_fraction_before_interpolation":
                valid_before,

            **background_current,
        }

        rejection_reason = ""

        if (
            sample_count
            < minimum_block_samples
        ):
            rejection_reason = (
                "analysis block too short"
            )

        elif (
            valid_before
            < minimum_valid_fraction
        ):
            rejection_reason = (
                "too many missing velocity samples"
            )

        if rejection_reason:

            statistics_records.append(
                {
                    **base_record,
                    "accepted":
                        False,

                    "rejection_reason":
                        rejection_reason,

                    "valid_fraction_after_interpolation":
                        np.nan,

                    **nan_results,
                }
            )

            continue


        # ----------------------------------------------------
        # Interpolate short gaps
        # ----------------------------------------------------

        cross_series = (
            interpolate_short_gaps(
                block[
                    cross_column
                ],
                maximum_interpolation_gap_samples,
            )
        )

        along_series = (
            interpolate_short_gaps(
                block[
                    along_column
                ],
                maximum_interpolation_gap_samples,
            )
        )

        finite_after = (
            cross_series.notna()
            & along_series.notna()
        )

        valid_after = (
            finite_after.mean()
        )

        if not finite_after.all():

            statistics_records.append(
                {
                    **base_record,
                    "accepted":
                        False,

                    "rejection_reason":
                        "missing values remain after interpolation",

                    "valid_fraction_after_interpolation":
                        valid_after,

                    **nan_results,
                }
            )

            continue


        # ----------------------------------------------------
        # Raw cell velocity
        # ----------------------------------------------------

        cross_raw = (
            cross_series.to_numpy(
                dtype=np.float64
            )
        )

        along_raw = (
            along_series.to_numpy(
                dtype=np.float64
            )
        )

        cell_mean_cross_current = float(
            np.mean(
                cross_raw
            )
        )

        cell_mean_along_current = float(
            np.mean(
                along_raw
            )
        )


        # ----------------------------------------------------
        # Frozen-signal QC
        # ----------------------------------------------------

        cross_std = float(
            np.std(
                cross_raw
            )
        )

        along_std = float(
            np.std(
                along_raw
            )
        )

        if (
            not np.isfinite(
                cross_std
            )
            or not np.isfinite(
                along_std
            )
            or (
                cross_std
                < minimum_velocity_std_m_s
            )
            or (
                along_std
                < minimum_velocity_std_m_s
            )
        ):

            statistics_records.append(
                {
                    **base_record,
                    "accepted":
                        False,

                    "rejection_reason":
                        "frozen or near-constant velocity signal",

                    "valid_fraction_after_interpolation":
                        valid_after,

                    **nan_results,
                }
            )

            continue


        # ----------------------------------------------------
        # Detrend
        # ----------------------------------------------------

        cross, cross_background = (
            polynomial_detrend(
                cross_raw,
                order=detrend_order,
            )
        )

        along, along_background = (
            polynomial_detrend(
                along_raw,
                order=detrend_order,
            )
        )

        cross_background_change = float(
            cross_background[-1]
            - cross_background[0]
        )

        along_background_change = float(
            along_background[-1]
            - along_background[0]
        )
        
        # ----------------------------------------------------
        # Welch autospectra
        # ----------------------------------------------------

        (
            frequency,
            cross_psd,
            along_psd,
            uv_csd,
        ) = calculate_autospectra(
            cross_shore=cross,
            alongshore=along,
            fs=fs,
            segment_seconds=welch_segment_seconds,
            overlap_fraction=overlap_fraction,
        )


        # ----------------------------------------------------
        # IG variance
        # ----------------------------------------------------

        cross_ig_variance = (
            integrate_spectral_band(
                frequency,
                cross_psd,
                ig_low_hz,
                ig_high_hz,
            )
        )

        along_ig_variance = (
            integrate_spectral_band(
                frequency,
                along_psd,
                ig_low_hz,
                ig_high_hz,
            )
        )
        
        ig_band = (
            (frequency >= ig_low_hz)
            & (frequency < ig_high_hz)
        )
        
        ig_uv_covariance = float(
            np.trapezoid(
                np.real(uv_csd[ig_band]),
                frequency[ig_band],
            )
        )


        # ----------------------------------------------------
        # Sea-swell variance
        # ----------------------------------------------------

        ss_upper_hz = min(
            ss_high_hz,
            0.5 * fs,
        )

        cross_ss_variance = (
            integrate_spectral_band(
                frequency,
                cross_psd,
                ss_low_hz,
                ss_upper_hz,
            )
        )

        along_ss_variance = (
            integrate_spectral_band(
                frequency,
                along_psd,
                ss_low_hz,
                ss_upper_hz,
            )
        )
        
        ss_band = (
            (frequency >= ss_low_hz)
            & (frequency < ss_upper_hz)
        )
        
        ss_uv_covariance = float(
            np.trapezoid(
                np.real(uv_csd[ss_band]),
                frequency[ss_band],
            )
        )

        # ----------------------------------------------------
        # Spectral QC
        # ----------------------------------------------------

        variances = np.asarray(
            [
                cross_ig_variance,
                along_ig_variance,
                cross_ss_variance,
                along_ss_variance,
            ],
            dtype=float,
        )

        if (
            not np.isfinite(
                variances
            ).all()
            or np.any(
                variances <= 0
            )
        ):

            statistics_records.append(
                {
                    **base_record,
                    "accepted":
                        False,

                    "rejection_reason":
                        "invalid or zero spectral variance",

                    "valid_fraction_after_interpolation":
                        valid_after,

                    **nan_results,
                }
            )

            continue
        
        # ============================================================
        # SAVE-READY DETRENDED VELOCITY TIME SERIES
        # For later pressure-velocity directional analysis
        # ============================================================
        
        directional_velocity_records.append(
            pd.DataFrame({
                "case_id": case_id,
                "case_name": case_label,
                "analysis_block_number": analysis_block_number,
                "source_segment_id": source_segment_id,
                "subblock_number": subblock_number,
                "cell_number": cell_number,
                "height_relative_to_frame_bottom_m": height_m,
                "time": block["time"].to_numpy(),
                "cross_velocity_detrended_m_s": cross,
                "along_velocity_detrended_m_s": along,
            })
        )

        # ----------------------------------------------------
        # RMS velocities
        # ----------------------------------------------------

        cross_ig_rms = float(
            np.sqrt(
                cross_ig_variance
            )
        )

        along_ig_rms = float(
            np.sqrt(
                along_ig_variance
            )
        )

        cross_ss_rms = float(
            np.sqrt(
                cross_ss_variance
            )
        )

        along_ss_rms = float(
            np.sqrt(
                along_ss_variance
            )
        )

        horizontal_ig_rms = float(
            np.hypot(
                cross_ig_rms,
                along_ig_rms,
            )
        )

        horizontal_ss_rms = float(
            np.hypot(
                cross_ss_rms,
                along_ss_rms,
            )
        )

        total_horizontal_ig_variance = float(
            cross_ig_variance
            + along_ig_variance
        )

        total_horizontal_ss_variance = float(
            cross_ss_variance
            + along_ss_variance
        )


        # ----------------------------------------------------
        # Statistics record
        # ----------------------------------------------------

        statistics_records.append(
            {
                **base_record,

                "accepted":
                    True,

                "rejection_reason":
                    "",

                "valid_fraction_after_interpolation":
                    valid_after,

                "cell_mean_cross_current_m_s":
                    cell_mean_cross_current,

                "cell_mean_along_current_m_s":
                    cell_mean_along_current,

                "cross_background_change_m_s":
                    cross_background_change,

                "along_background_change_m_s":
                    along_background_change,

                "cross_ig_variance_m2_s2":
                    cross_ig_variance,

                "along_ig_variance_m2_s2":
                    along_ig_variance,

                "total_horizontal_ig_variance_m2_s2":
                    total_horizontal_ig_variance,

                "cross_ss_variance_m2_s2":
                    cross_ss_variance,

                "along_ss_variance_m2_s2":
                    along_ss_variance,

                "total_horizontal_ss_variance_m2_s2":
                    total_horizontal_ss_variance,
                
                "ig_uv_covariance_m2_s2":
                    ig_uv_covariance,

                "ss_uv_covariance_m2_s2":
                    ss_uv_covariance,

                "cross_ig_rms_m_s":
                    cross_ig_rms,

                "along_ig_rms_m_s":
                    along_ig_rms,

                "horizontal_ig_rms_m_s":
                    horizontal_ig_rms,

                "cross_ss_rms_m_s":
                    cross_ss_rms,

                "along_ss_rms_m_s":
                    along_ss_rms,

                "horizontal_ss_rms_m_s":
                    horizontal_ss_rms,
            }
        )


        # ----------------------------------------------------
        # Full spectral record
        # ----------------------------------------------------

        spectral_records.append(
            pd.DataFrame(
                {
                    "analysis_block_number":
                        analysis_block_number,

                    "source_segment_id":
                        source_segment_id,

                    "subblock_number":
                        subblock_number,

                    "cell_number":
                        cell_number,

                    "height_relative_to_frame_bottom_m":
                        height_m,

                    "start_time":
                        start_time,

                    "end_time":
                        end_time,

                    "frequency_hz":
                        frequency,

                    "cross_psd_m2_s2_hz":
                        cross_psd,

                    "along_psd_m2_s2_hz":
                        along_psd,
                }
            )
        )


# ============================================================
# BUILD RESULTS
# ============================================================

statistics = pd.DataFrame(
    statistics_records
)

spectra = (
    pd.concat(
        spectral_records,
        ignore_index=True,
    )
    if spectral_records
    else pd.DataFrame()
)

accepted = (
    statistics.loc[
        statistics["accepted"]
    ]
    .copy()
    .sort_values(
        [
            "cell_number",
            "mid_time",
        ]
    )
)

if accepted.empty:
    raise RuntimeError(
        "No selected cell/burst combinations passed QC."
    )

print(
    "\nAccepted bursts by cell:"
)

print(
    accepted.groupby(
        [
            "cell_number",
            "height_relative_to_frame_bottom_m",
        ]
    ).size()
)

print(
    "\nCurrent profile cell count:"
)

print(
    accepted[
        "current_profile_cell_count"
    ].describe()
)

directional_velocity = (
    pd.concat(directional_velocity_records, ignore_index=True)
    if directional_velocity_records
    else pd.DataFrame()
)
# ============================================================
# OPTIONAL SAVE
# ============================================================

# statistics_file = (
#     output_folder
#     / f"{case_id}_selected_cells_spectral_statistics.csv"
# )
#
# spectra_file = (
#     output_folder
#     / f"{case_id}_selected_cells_velocity_spectra.pkl"
# )
#
# segment_file = (
#     output_folder
#     / f"{case_id}_segment_summary.csv"
# )
#
# statistics.to_csv(
#     statistics_file,
#     index=False,
# )
#
# spectra.to_pickle(
#     spectra_file
# )
#
# segment_summary.to_csv(
#     segment_file,
#     index=False,
# )

# ============================================================
# SAVE DETRENDED VELOCITY FOR LATER PRESSURE-VELOCITY ANALYSIS
# Shared file for F1 and F3
# ============================================================

direction_file = (
    output_folder
    / "all_cases_ADCP_directional_velocity.parquet"
)

if direction_file.exists():
    old = pd.read_parquet(direction_file)

    # Replace this case when rerunning, avoiding duplicates
    old = old.loc[old["case_id"] != case_id]

    directional_velocity = pd.concat(
        [old, directional_velocity],
        ignore_index=True,
    )

directional_velocity = directional_velocity.sort_values(
    ["case_id", "time", "cell_number"]
)

directional_velocity.to_parquet(
    direction_file,
    index=False,
)

print(
    "\nSaved directional velocity time series:\n"
    f"{direction_file}"
)

# ============================================================
# SAVE BURST-LEVEL CURRENT + WAVE-VELOCITY STATISTICS
# Shared CSV for F1 and F3
# ============================================================

burst_file = (
    output_folder
    / "all_cases_ADCP_current_wave_axis.csv"
)

save_columns = [
    "analysis_block_number",
    "source_segment_id",
    "subblock_number",
    "start_time",
    "end_time",
    "mid_time",
    "cell_number",
    "height_relative_to_frame_bottom_m",

    # Full-profile background current
    "depth_avg_cross_current_m_s",
    "depth_avg_along_current_m_s",
    "depth_avg_current_speed_m_s",
    "depth_avg_current_direction_deg",
    "current_profile_cell_count",

    # IG variance, covariance and RMS
    "cross_ig_variance_m2_s2",
    "along_ig_variance_m2_s2",
    "ig_uv_covariance_m2_s2",
    "total_horizontal_ig_variance_m2_s2",
    "cross_ig_rms_m_s",
    "along_ig_rms_m_s",
    "horizontal_ig_rms_m_s",

    # Sea-swell variance, covariance and RMS
    "cross_ss_variance_m2_s2",
    "along_ss_variance_m2_s2",
    "ss_uv_covariance_m2_s2",
    "total_horizontal_ss_variance_m2_s2",
    "cross_ss_rms_m_s",
    "along_ss_rms_m_s",
    "horizontal_ss_rms_m_s",
]

burst_data = accepted[save_columns].copy()

# Case information
burst_data.insert(0, "case_name", case_label)
burst_data.insert(0, "case_id", case_id)

# Processing settings, useful when comparing/reprocessing cases
burst_data["ig_low_hz"] = ig_low_hz
burst_data["ig_high_hz"] = ig_high_hz
burst_data["ss_low_hz"] = ss_low_hz
burst_data["ss_high_hz"] = ss_high_hz
burst_data["detrend_order"] = detrend_order
burst_data["welch_segment_seconds"] = welch_segment_seconds
burst_data["overlap_fraction"] = overlap_fraction

# Replace this case if the shared file already exists
if burst_file.exists():
    old = pd.read_csv(
        burst_file,
        parse_dates=["start_time", "end_time", "mid_time"],
    )

    old = old.loc[
        old["case_id"] != case_id
    ]

    burst_data = pd.concat(
        [old, burst_data],
        ignore_index=True,
    )

burst_data = burst_data.sort_values(
    ["case_id", "mid_time", "cell_number"]
)

burst_data.to_csv(
    burst_file,
    index=False,
)

print(
    "\nSaved burst-level current/wave statistics:\n"
    f"{burst_file}"
)

print(
    burst_data.groupby("case_id").size()
)

# ============================================================
# PLOTTING CELLS
# ============================================================

plot_cells = spectral_cells


# ============================================================
# COMMON BACKGROUND-CURRENT DATA
# ============================================================

current_cols = [
    "depth_avg_cross_current_m_s",
    "depth_avg_along_current_m_s",
    "depth_avg_current_speed_m_s",
    "depth_avg_current_direction_deg",
]

current_data = (
    accepted[
        [
            "analysis_block_number",
            "mid_time",
            *current_cols,
        ]
    ]
    .drop_duplicates(
        "analysis_block_number"
    )
    .sort_values(
        "mid_time"
    )
    .reset_index(
        drop=True
    )
)

gap = (
    current_data[
        "mid_time"
    ]
    .diff()
    .dt.total_seconds()
    > plot_gap_seconds
)

current_plot = (
    current_data.copy()
)

current_plot.loc[
    gap,
    current_cols,
] = np.nan


# ============================================================
# FIGURE 1: FULL-PROFILE BACKGROUND CURRENT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 7),
    sharex=True,
)

plot_info = [
    (
        "depth_avg_cross_current_m_s",
        r"$U_{\mathrm{current}}$ (m/s)",
        "Depth-averaged cross-shore current",
    ),
    (
        "depth_avg_along_current_m_s",
        r"$V_{\mathrm{current}}$ (m/s)",
        "Depth-averaged alongshore current",
    ),
    (
        "depth_avg_current_speed_m_s",
        r"$|\mathbf{U}_{\mathrm{current}}|$ (m/s)",
        "Depth-averaged current speed",
    ),
]

for ax, (
    column,
    ylabel,
    label,
) in zip(
    axes,
    plot_info,
):

    ax.plot(
        current_plot[
            "mid_time"
        ],
        current_plot[
            column
        ],
        label=label,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right"
    )

axes[-1].set_ylim(
    bottom=0
)

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    f"Full-profile depth-averaged background current "
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)

plt.show()


# ============================================================
# FIGURE 2: IG RMS + CURRENT SPEED, HIGHEST SELECTED CELL
# ============================================================

highest_cell = max(
    spectral_cells,
    key=lambda cell:
        cell[
            "height_relative_to_frame_bottom_m"
        ],
)

cell_number = (
    highest_cell[
        "cell_number"
    ]
)

height_m = (
    highest_cell[
        "height_relative_to_frame_bottom_m"
    ]
)

cell_data = (
    accepted.loc[
        accepted[
            "cell_number"
        ]
        == cell_number
    ]
    .sort_values(
        "mid_time"
    )
    .copy()
)

plot_cols = {
    "cross_ig_plot":
        "cross_ig_rms_m_s",

    "along_ig_plot":
        "along_ig_rms_m_s",

    "horizontal_ig_plot":
        "horizontal_ig_rms_m_s",

    "current_speed_plot":
        "depth_avg_current_speed_m_s",
}

for (
    new_column,
    source_column,
) in plot_cols.items():

    cell_data[
        new_column
    ] = cell_data[
        source_column
    ]

gap = (
    cell_data[
        "mid_time"
    ]
    .diff()
    .dt.total_seconds()
    > plot_gap_seconds
)

cell_data.loc[
    gap,
    list(
        plot_cols
    ),
] = np.nan

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10),
    sharex=True,
)

plot_info = [
    (
        "cross_ig_plot",
        r"$U_{\mathrm{RMS,IG}}$ (m/s)",
    ),
    (
        "along_ig_plot",
        r"$V_{\mathrm{RMS,IG}}$ (m/s)",
    ),
    (
        "horizontal_ig_plot",
        r"$|\mathbf{U}_{\mathrm{RMS,IG}}|$ (m/s)",
    ),
    (
        "current_speed_plot",
        r"$U_{\mathrm{current}}$ (m/s)",
    ),
]

for ax, (
    column,
    ylabel,
) in zip(
    axes,
    plot_info,
):

    ax.plot(
        cell_data[
            "mid_time"
        ],
        cell_data[
            column
        ],
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        alpha=0.25,
    )

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    "IG velocity RMS and full-profile current speed\n"
    f"Cell {cell_number}, "
    f"z={height_m:.3f} m "
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)

plt.show()

# SS
plot_cols_SS = {
    "cross_ss_plot":
        "cross_ss_rms_m_s",

    "along_ss_plot":
        "along_ss_rms_m_s",

    "horizontal_ss_plot":
        "horizontal_ss_rms_m_s",

    "current_speed_plot":
        "depth_avg_current_speed_m_s",
}

for (
    new_column,
    source_column,
) in plot_cols_SS.items():

    cell_data[
        new_column
    ] = cell_data[
        source_column
    ]

gap = (
    cell_data[
        "mid_time"
    ]
    .diff()
    .dt.total_seconds()
    > plot_gap_seconds
)

cell_data.loc[
    gap,
    list(
        plot_cols_SS
    ),
] = np.nan

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10),
    sharex=True,
)

plot_info = [
    (
        "cross_ss_plot",
        r"$U_{\mathrm{RMS,SS}}$ (m/s)",
    ),
    (
        "along_ss_plot",
        r"$V_{\mathrm{RMS,SS}}$ (m/s)",
    ),
    (
        "horizontal_ss_plot",
        r"$|\mathbf{U}_{\mathrm{RMS,SS}}|$ (m/s)",
    ),
    (
        "current_speed_plot",
        r"$U_{\mathrm{current}}$ (m/s)",
    ),
]

for ax, (
    column,
    ylabel,
) in zip(
    axes,
    plot_info,
):

    ax.plot(
        cell_data[
            "mid_time"
        ],
        cell_data[
            column
        ],
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        alpha=0.25,
    )

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    "SS velocity RMS and full-profile current speed\n"
    f"Cell {cell_number}, "
    f"z={height_m:.3f} m "
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)

plt.show()


# ============================================================
# FIGURE 3: VERTICAL CONSISTENCY OF IG RMS
# ============================================================
fig, axes = plt.subplots(
    1 + len(plot_cells),
    1,
    figsize=(12, 11),
    sharex=True,
)

axes[0].plot(
    current_plot[
        "mid_time"
    ],
    current_plot[
        "depth_avg_current_speed_m_s"
    ],
)

axes[0].set_ylabel(
    r"$U_{\mathrm{current}}$"
    "\n"
    r"$(\mathrm{m/s})$"
)

axes[0].set_ylim(
    bottom=0
)

axes[0].grid(
    True,
    alpha=0.25,
)

for ax, cell in zip(
    axes[1:],
    plot_cells,
):

    cell_number = (
        cell[
            "cell_number"
        ]
    )

    height_m = (
        cell[
            "height_relative_to_frame_bottom_m"
        ]
    )

    d = (
        accepted.loc[
            accepted[
                "cell_number"
            ]
            == cell_number
        ]
        .sort_values(
            "mid_time"
        )
        .copy()
    )

    gap = (
        d[
            "mid_time"
        ]
        .diff()
        .dt.total_seconds()
        > plot_gap_seconds
    )

    d[
        "ig_plot"
    ] = d[
        "horizontal_ig_rms_m_s"
    ]

    d.loc[
        gap,
        "ig_plot",
    ] = np.nan

    ax.plot(
        d[
            "mid_time"
        ],
        d[
            "ig_plot"
        ],
    )

    ax.set_ylabel(
        r"$|\mathbf{U}_{\mathrm{RMS,IG}}|$"
        "\n"
        r"$(\mathrm{m/s})$"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.88,
        f"Cell {cell_number}, "
        f"z={height_m:.3f} m",
        transform=
            ax.transAxes,
        ha="left",
        va="top",
    )

ig_max = (
    accepted[
        "horizontal_ig_rms_m_s"
    ]
    .max()
)

if np.isfinite(
    ig_max
):
    for ax in axes[1:]:
        ax.set_ylim(
            0,
            ig_max * 1.05,
        )

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    "Full-profile current speed and vertical consistency "
    "of IG velocity RMS\n"
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)

plt.show()


# ============================================================
# FIGURE 4: IG RMS VS CURRENT SPEED, CURRENT-DIRECTION SPLIT
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()

end_time = pd.Timestamp(
    "2018-05-15 00:00:00",
    tz="UTC",
)

for ax, cell in zip(
    axes,
    plot_cells,
):

    cell_number = (
        cell[
            "cell_number"
        ]
    )

    height_m = (
        cell[
            "height_relative_to_frame_bottom_m"
        ]
    )

    d = accepted.loc[
        (
            accepted[
                "cell_number"
            ]
            == cell_number
        )
        & (
            accepted[
                "mid_time"
            ]
            < end_time
        ),
        [
            "depth_avg_current_speed_m_s",
            "depth_avg_along_current_m_s",
            "horizontal_ig_rms_m_s",
        ],
    ].dropna()

    positive = (
        d[
            "depth_avg_along_current_m_s"
        ]
        > 0
    )

    negative = (
        d[
            "depth_avg_along_current_m_s"
        ]
        < 0
    )

    ax.scatter(
        d.loc[
            positive,
            "depth_avg_current_speed_m_s",
        ],
        d.loc[
            positive,
            "horizontal_ig_rms_m_s",
        ],
        s=15,
        alpha=0.6,
        label=
            r"$V_{\mathrm{current}}>0$",
    )

    ax.scatter(
        d.loc[
            negative,
            "depth_avg_current_speed_m_s",
        ],
        d.loc[
            negative,
            "horizontal_ig_rms_m_s",
        ],
        s=15,
        alpha=0.6,
        label=
            r"$V_{\mathrm{current}}<0$",
    )

    ax.set_title(
        f"Cell {cell_number}, "
        f"z={height_m:.3f} m"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

for ax in axes[2:]:
    ax.set_xlabel(
        r"$U_{\mathrm{current}}$ (m/s)"
    )

for ax in axes[::2]:
    ax.set_ylabel(
        r"$|\mathbf{U}_{\mathrm{RMS,IG}}|$ (m/s)"
    )

fig.suptitle(
    "IG-band velocity RMS vs full-profile current speed\n"
    f"Before 2018-05-15 00:00 UTC "
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95,
    ]
)

plt.show()


# ============================================================
# FIGURE 5: VERTICAL CONSISTENCY OF SEA-SWELL RMS
# ============================================================

fig, axes = plt.subplots(
    1 + len(plot_cells),
    1,
    figsize=(12, 11),
    sharex=True,
)

axes[0].plot(
    current_plot[
        "mid_time"
    ],
    current_plot[
        "depth_avg_current_speed_m_s"
    ],
)

axes[0].set_ylabel(
    r"$U_{\mathrm{current}}$"
    "\n"
    r"$(\mathrm{m/s})$"
)

axes[0].set_ylim(
    bottom=0
)

axes[0].grid(
    True,
    alpha=0.25,
)

for ax, cell in zip(
    axes[1:],
    plot_cells,
):

    cell_number = (
        cell[
            "cell_number"
        ]
    )

    height_m = (
        cell[
            "height_relative_to_frame_bottom_m"
        ]
    )

    d = (
        accepted.loc[
            accepted[
                "cell_number"
            ]
            == cell_number
        ]
        .sort_values(
            "mid_time"
        )
        .copy()
    )

    gap = (
        d[
            "mid_time"
        ]
        .diff()
        .dt.total_seconds()
        > plot_gap_seconds
    )

    d[
        "ss_plot"
    ] = d[
        "horizontal_ss_rms_m_s"
    ]

    d.loc[
        gap,
        "ss_plot",
    ] = np.nan

    ax.plot(
        d[
            "mid_time"
        ],
        d[
            "ss_plot"
        ],
    )

    ax.set_ylabel(
        r"$|\mathbf{U}_{\mathrm{RMS,SS}}|$"
        "\n"
        r"$(\mathrm{m/s})$"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.88,
        f"Cell {cell_number}, "
        f"z={height_m:.3f} m",
        transform=
            ax.transAxes,
        ha="left",
        va="top",
    )

ss_max = (
    accepted[
        "horizontal_ss_rms_m_s"
    ]
    .max()
)

if np.isfinite(
    ss_max
):
    for ax in axes[1:]:
        ax.set_ylim(
            0,
            ss_max * 1.05,
        )

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    "Full-profile current speed and vertical consistency "
    "of SS velocity RMS\n"
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)

plt.show()


# ============================================================
# FIGURE 6: IG / SS VELOCITY-VARIANCE RATIO
# ============================================================

highest_cell = max(
    spectral_cells,
    key=lambda cell:
        cell[
            "height_relative_to_frame_bottom_m"
        ],
)

cell_number = (
    highest_cell[
        "cell_number"
    ]
)

height_m = (
    highest_cell[
        "height_relative_to_frame_bottom_m"
    ]
)

d = (
    accepted.loc[
        accepted[
            "cell_number"
        ]
        == cell_number
    ]
    .sort_values(
        "mid_time"
    )
    .copy()
)

d["R_u"] = (
    d[
        "horizontal_ig_rms_m_s"
    ] ** 2
    /
    d[
        "horizontal_ss_rms_m_s"
    ] ** 2
)

d.loc[
    ~np.isfinite(
        d["R_u"]
    ),
    "R_u",
] = np.nan

gap = (
    d[
        "mid_time"
    ]
    .diff()
    .dt.total_seconds()
    > plot_gap_seconds
)

d.loc[
    gap,
    [
        "R_u",
        "depth_avg_current_speed_m_s",
    ],
] = np.nan

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 6),
    sharex=True,
)

axes[0].plot(
    d[
        "mid_time"
    ],
    d[
        "R_u"
    ],
)

axes[0].set_ylabel(
    r"$R_u$"
)

axes[0].grid(
    True,
    alpha=0.25,
)

axes[1].plot(
    d[
        "mid_time"
    ],
    d[
        "depth_avg_current_speed_m_s"
    ],
)

axes[1].set_ylabel(
    r"$U_{\mathrm{current}}$ (m/s)"
)

axes[1].set_xlabel(
    "Time"
)

axes[1].set_ylim(
    bottom=0
)

axes[1].grid(
    True,
    alpha=0.25,
)

fig.suptitle(
    r"$R_u="
    r"|\mathbf{U}_{\mathrm{RMS,IG}}|^2/"
    r"|\mathbf{U}_{\mathrm{RMS,SS}}|^2$"
    " and full-profile current speed"
    f"\nCell {cell_number}, "
    f"z={height_m:.3f} m "
    f"({case_label})"
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

plt.show()


# # ============================================================
# # SELECT MOST ENERGETIC IG BURST ACROSS THE PROFILE
# # ============================================================

# burst_energy = (
#     accepted.groupby("analysis_block_number", as_index=False)
#     .agg(
#         start_time=("start_time", "first"),
#         end_time=("end_time", "first"),
#         accepted_cell_count=("cell_number", "nunique"),
#         profile_mean_total_ig_variance=("total_horizontal_ig_variance_m2_s2", "mean"),
#         profile_median_total_ig_variance=("total_horizontal_ig_variance_m2_s2", "median"),
#     )
# )

# minimum_cells_for_energetic_selection = max(2, int(np.ceil(0.75 * len(cells))))

# energetic_candidates = burst_energy.loc[
#     burst_energy["accepted_cell_count"] >= minimum_cells_for_energetic_selection
# ].copy()

# if energetic_candidates.empty:
#     energetic_candidates = burst_energy.copy()

# most_energetic_row = energetic_candidates.loc[
#     energetic_candidates["profile_mean_total_ig_variance"].idxmax()
# ]

# most_energetic_block_number = int(most_energetic_row["analysis_block_number"])

# most_energetic_spectra = spectra.loc[
#     spectra["analysis_block_number"] == most_energetic_block_number
# ].copy()


# # ============================================================
# # MOST ENERGETIC BURST SPECTRA: 13 SUBPLOTS
# # ============================================================

# fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
# legend_handles = None
# legend_labels = None

# for axis, cell in zip(axes, plot_cells):
#     cell_number = cell["cell_number"]
#     height_m = cell["height_relative_to_frame_bottom_m"]

#     cell_spectrum = most_energetic_spectra.loc[
#         (most_energetic_spectra["cell_number"] == cell_number)
#         & (most_energetic_spectra["frequency_hz"] > 0)
#     ].sort_values("frequency_hz")

#     axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

#     if cell_spectrum.empty:
#         axis.text(0.5, 0.5, "No accepted spectrum", ha="center", va="center", transform=axis.transAxes)
#         continue

#     axis.loglog(
#         cell_spectrum["frequency_hz"],
#         cell_spectrum["cross_psd_m2_s2_hz"],
#         label="Cross-shore",
#     )
#     axis.loglog(
#         cell_spectrum["frequency_hz"],
#         cell_spectrum["along_psd_m2_s2_hz"],
#         label="Alongshore",
#     )
#     axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
#     axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
#     axis.set_xlabel("Frequency (Hz)")
#     axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
#     axis.grid(True, which="both", alpha=0.25)
#     axis.set_ylim(1e-4, 3)

#     if legend_handles is None:
#         legend_handles, legend_labels = axis.get_legend_handles_labels()

# fig.suptitle(
#     "Most energetic profile-wide IG burst\n"
#     f"{case_label}: {most_energetic_row['start_time']} to {most_energetic_row['end_time']}",
#     y=0.995,
# )

# if legend_handles:
#     fig.legend(
#         legend_handles,
#         legend_labels,
#         loc="upper center",
#         ncol=4,
#         bbox_to_anchor=(0.5, 0.955),
#     )

# fig.tight_layout(rect=[0, 0, 1, 0.91])
# plt.show()


# # ============================================================
# # MEDIAN SPECTRA: 13 SUBPLOTS
# # ============================================================

# median_spectra = (
#     spectra.groupby(
#         ["cell_number", "height_relative_to_frame_bottom_m", "frequency_hz"],
#         as_index=False,
#     )
#     .agg(
#         cross_psd_median=("cross_psd_m2_s2_hz", "median"),
#         along_psd_median=("along_psd_m2_s2_hz", "median"),
#         cross_psd_q25=("cross_psd_m2_s2_hz", lambda values: values.quantile(0.25)),
#         cross_psd_q75=("cross_psd_m2_s2_hz", lambda values: values.quantile(0.75)),
#         along_psd_q25=("along_psd_m2_s2_hz", lambda values: values.quantile(0.25)),
#         along_psd_q75=("along_psd_m2_s2_hz", lambda values: values.quantile(0.75)),
#     )
# )

# fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
# legend_handles = None
# legend_labels = None

# for axis, cell in zip(axes, plot_cells):
#     cell_number = cell["cell_number"]
#     height_m = cell["height_relative_to_frame_bottom_m"]

#     plot_data = median_spectra.loc[
#         (median_spectra["cell_number"] == cell_number)
#         & (median_spectra["frequency_hz"] > 0)
#     ].sort_values("frequency_hz")

#     axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

#     if plot_data.empty:
#         axis.text(0.5, 0.5, "No accepted spectra", ha="center", va="center", transform=axis.transAxes)
#         continue

#     axis.loglog(plot_data["frequency_hz"], plot_data["cross_psd_median"], label="Cross-shore median")
#     axis.loglog(plot_data["frequency_hz"], plot_data["along_psd_median"], label="Alongshore median")
#     axis.fill_between(
#         plot_data["frequency_hz"],
#         plot_data["cross_psd_q25"],
#         plot_data["cross_psd_q75"],
#         alpha=0.12,
#     )
#     axis.fill_between(
#         plot_data["frequency_hz"],
#         plot_data["along_psd_q25"],
#         plot_data["along_psd_q75"],
#         alpha=0.12,
#     )
#     axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
#     axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
#     axis.set_ylim(1e-4, 1e-1)
#     axis.set_xlabel("Frequency (Hz)")
#     axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
#     axis.grid(True, which="both", alpha=0.25)
#     axis.set_ylim(1e-4, 1e-1)

#     if legend_handles is None:
#         legend_handles, legend_labels = axis.get_legend_handles_labels()

# fig.suptitle(f"Median ADCP-HR velocity autospectra ({case_label})", y=0.995)

# if legend_handles:
#     fig.legend(
#         legend_handles,
#         legend_labels,
#         loc="upper center",
#         ncol=4,
#         bbox_to_anchor=(0.5, 0.965),
#     )

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.show()


# # ============================================================
# # MEAN SPECTRA: 13 SUBPLOTS
# # ============================================================

# mean_spectra = (
#     spectra.groupby(
#         ["cell_number", "height_relative_to_frame_bottom_m", "frequency_hz"],
#         as_index=False,
#     )
#     .agg(
#         cross_psd_mean=("cross_psd_m2_s2_hz", "mean"),
#         along_psd_mean=("along_psd_m2_s2_hz", "mean"),
#         cross_psd_std=("cross_psd_m2_s2_hz", "std"),
#         along_psd_std=("along_psd_m2_s2_hz", "std"),
#     )
# )

# fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
# legend_handles = None
# legend_labels = None

# for axis, cell in zip(axes, plot_cells):
#     cell_number = cell["cell_number"]
#     height_m = cell["height_relative_to_frame_bottom_m"]

#     plot_data = mean_spectra.loc[
#         (mean_spectra["cell_number"] == cell_number)
#         & (mean_spectra["frequency_hz"] > 0)
#     ].sort_values("frequency_hz")

#     axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

#     if plot_data.empty:
#         axis.text(0.5, 0.5, "No accepted spectra", ha="center", va="center", transform=axis.transAxes)
#         continue

#     axis.loglog(plot_data["frequency_hz"], plot_data["cross_psd_mean"], label="Cross-shore mean")
#     axis.loglog(plot_data["frequency_hz"], plot_data["along_psd_mean"], label="Alongshore mean")
#     axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
#     axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
#     axis.set_ylim(2e-4, 8e-2)
#     axis.set_xlabel("Frequency (Hz)")
#     axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
#     axis.grid(True, which="both", alpha=0.25)

#     if legend_handles is None:
#         legend_handles, legend_labels = axis.get_legend_handles_labels()

# fig.suptitle(f"Mean ADCP-HR velocity autospectra ({case_label})", y=0.995)

# if legend_handles:
#     fig.legend(
#         legend_handles,
#         legend_labels,
#         loc="upper center",
#         ncol=4,
#         bbox_to_anchor=(0.5, 0.965),
#     )

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.show()

