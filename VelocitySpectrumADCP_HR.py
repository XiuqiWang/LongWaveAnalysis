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
from scipy.signal import welch


# ============================================================
# USER SETTINGS
# ============================================================

input_file = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
    r"\DVN_F1_ADCP_HR_rotated.parquet"
)

# Cell metadata created by the ADCP extraction script.
# cell_metadata_file = Path(
#     r"C:\dev\Python\LongWaveAnalysis\Processed"
#     r"\DVN_F4_ADCP_cell_metadata.csv"
# )

case_id = "DVN_F1_ADCP-HR"
case_label = "DVN F1 ADCP-HR"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)
output_folder.mkdir(parents=True, exist_ok=True)

# A new natural segment starts when the timestamp interval is larger
# than this value. The script also checks for nonpositive timestamp steps.
gap_threshold_seconds = 2.0

# Welch settings.
welch_segment_seconds = 512.0
overlap_fraction = 0.5

# Frequency bands.
ig_low_hz = 0.005
ig_high_hz = 0.05

ss_low_hz = 0.05
ss_high_hz = 0.50

# A burst/cell combination is accepted only when at least this fraction
# of both cross-shore and alongshore samples is present.
minimum_valid_fraction = 0.98

# Accept a final analysis block when it contains at least this fraction
# of the automatically detected nominal burst length.
minimum_block_fraction = 0.95

# Polynomial order used to remove the slowly varying background current.
detrend_order = 2

# Interpolate only short internal gaps. Long corrupted sections remain NaN
# and cause that cell/burst combination to be rejected.
maximum_interpolation_gap_seconds = 2.0

# Break plotted lines when accepted bursts are separated by more than this.
plot_gap_seconds = 3600.0


# ============================================================
# FUNCTIONS
# ============================================================

def identify_velocity_columns(columns):
    """
    Identify paired ADCP-HR cross-shore and alongshore columns.

    Expected examples:
        cross_shore_cell01_id01_zp0123mm_m_s
        alongshore_cell01_id01_zp0123mm_m_s
        cross_shore_cell01_id01_zm0030mm_m_s
    """

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

        key = (
            cell_number,
            cell_id,
            height_mm,
        )

        found.setdefault(
            key,
            {
                "cell_number": cell_number,
                "cell_id": cell_id,
                "z_bin_m": height_mm / 1000.0,
            },
        )

        found[key][component] = column

    cells = []

    for key in sorted(found):
        item = found[key]

        if (
            "cross_shore" in item
            and "alongshore" in item
        ):
            cells.append(item)

    if not cells:
        raise KeyError(
            "No paired ADCP-HR cross-shore/"
            "alongshore columns were found."
        )

    return cells


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

    # Use the lower part of the interval distribution so that natural
    # inter-burst gaps do not influence the estimate.
    cutoff = np.nanpercentile(positive, 25)
    regular = positive[positive <= cutoff * 1.5]

    if regular.size == 0:
        regular = positive

    median_dt = float(np.nanmedian(regular))

    if median_dt <= 0:
        raise ValueError(
            "Detected a nonpositive sampling interval."
        )

    return 1.0 / median_dt, median_dt


def identify_continuous_segments(
    df,
    gap_threshold_seconds,
):
    """Assign a new segment ID at large or nonpositive timestamp steps."""
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

    output["segment_id"] = new_segment.cumsum()

    return output


def split_segment_into_analysis_blocks(
    segment,
    nominal_block_samples,
    minimum_block_samples,
):
    """Split one continuous segment into nominal-length analysis blocks."""
    blocks = []
    number_of_samples = len(segment)

    start = 0
    subblock_number = 0

    while (
        number_of_samples - start
        >= nominal_block_samples
    ):
        stop = start + nominal_block_samples

        block = segment.iloc[start:stop].copy()
        block["subblock_number"] = subblock_number
        blocks.append(block)

        start = stop
        subblock_number += 1

    remainder = segment.iloc[start:].copy()

    if len(remainder) >= minimum_block_samples:
        remainder["subblock_number"] = subblock_number
        blocks.append(remainder)

    return blocks


def polynomial_detrend(values, order=2):
    """Remove a polynomial background from a finite one-dimensional array."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            "Input must be one-dimensional."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains non-finite values."
        )

    x = np.linspace(-1.0, 1.0, len(values))

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


def interpolate_short_gaps(series, limit_samples):
    """
    Interpolate internal gaps no longer than limit_samples.

    Leading/trailing gaps and longer internal gaps remain NaN.
    """
    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    return series.interpolate(
        method="linear",
        limit=limit_samples,
        limit_area="inside",
    )


def calculate_autospectra(
    cross_shore,
    alongshore,
    fs,
    segment_seconds,
    overlap_fraction,
):
    """Calculate one-sided Welch autospectra for one ADCP cell."""
    cross = np.asarray(
        cross_shore,
        dtype=np.float64,
    )
    along = np.asarray(
        alongshore,
        dtype=np.float64,
    )

    if cross.shape != along.shape:
        raise ValueError(
            "Cross-shore and alongshore arrays differ in shape."
        )

    if cross.ndim != 1:
        raise ValueError(
            "Velocity arrays must be one-dimensional."
        )

    if (
        not np.isfinite(cross).all()
        or not np.isfinite(along).all()
    ):
        raise ValueError(
            "Velocity arrays contain missing values."
        )

    nperseg = min(
        int(round(segment_seconds * fs)),
        len(cross),
    )

    if nperseg < 16:
        raise ValueError(
            "Analysis block is too short for Welch analysis."
        )

    noverlap = min(
        int(round(overlap_fraction * nperseg)),
        nperseg - 1,
    )

    frequency, cross_psd = welch(
        cross,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    frequency_along, along_psd = welch(
        along,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    if not np.allclose(
        frequency,
        frequency_along,
    ):
        raise RuntimeError(
            "Cross-shore and alongshore frequency grids differ."
        )

    return frequency, cross_psd, along_psd


def integrate_spectral_band(
    frequency,
    psd,
    lower_hz,
    upper_hz,
):
    """Integrate a PSD over [lower_hz, upper_hz)."""
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


def attach_cell_heights(cells, metadata_file):
    """Attach authoritative z_measures heights from the metadata CSV."""
    if not metadata_file.exists():
        for cell in cells:
            cell["height_relative_to_frame_bottom_m"] = (
                cell["height_cm_from_column"] / 100.0
            )
        return cells

    metadata = pd.read_csv(metadata_file)

    required = {
        "cell_number",
        "height_relative_to_frame_bottom_m",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise KeyError(
            f"Cell metadata lacks columns: {sorted(missing)}"
        )

    height_lookup = dict(
        zip(
            metadata["cell_number"].astype(int),
            metadata["height_relative_to_frame_bottom_m"].astype(float),
        )
    )

    for cell in cells:
        cell_number = cell["cell_number"]

        cell["height_relative_to_frame_bottom_m"] = (
            height_lookup.get(
                cell_number,
                cell["height_cm_from_column"] / 100.0,
            )
        )

    return cells


def make_plot_label(cell_number, height_m):
    return (
        f"Cell {cell_number} "
        f"(z={height_m:.2f} m)"
    )


# ============================================================
# READ DATA AND DISCOVER CELLS
# ============================================================

print("Reading:")
print(input_file)

# Read only the Parquet schema first.
# This does not load the large data arrays.
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile(input_file)

available_columns = (
    parquet_file.schema_arrow.names
)

if "time" not in available_columns:
    raise KeyError(
        "The Parquet file does not contain a time column."
    )

cells = identify_velocity_columns(
    available_columns
)

print("\nDetected ADCP-HR cells:")

for cell in cells:
    print(
        cell["cell_number"],
        cell["cell_id"],
        cell["z_bin_m"],
        cell["cross_shore"],
        cell["alongshore"],
    )

required_columns = ["time"]

for cell in cells:
    required_columns.extend(
        [
            cell["cross_shore"],
            cell["alongshore"],
        ]
    )

# Load only time and the 26 rotated velocity columns.
df = pd.read_parquet(
    input_file,
    columns=required_columns,
)
# cells = attach_cell_heights(
#     cells,
#     cell_metadata_file,
# )
for cell in cells:
    cell["height_relative_to_frame_bottom_m"] = (
        cell["z_bin_m"]
    )

print("\nDetected ADCP cells:")

for cell in cells:
    print(
        cell["cell_number"],
        cell["height_relative_to_frame_bottom_m"],
        cell["cross_shore"],
        cell["alongshore"],
    )


df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
    errors="coerce",
)

df = (
    df.dropna(subset=["time"])
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

fs, regular_dt_seconds = detect_sampling_frequency(
    df["time"]
)

print("\nDetected sampling:")
print("dt =", regular_dt_seconds, "s")
print("fs =", fs, "Hz")

print("\nRecord:")
print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows: ", len(df))

# # ============================================================
# # IDENTIFY NATURAL BURSTS
# # ============================================================

df = identify_continuous_segments(
    df,
    gap_threshold_seconds=gap_threshold_seconds,
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

segment_summary["duration_minutes"] = (
    segment_summary["sample_count"]
    / fs
    / 60.0
)

segment_length_counts = (
    segment_summary["sample_count"]
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

print("\nContinuous segments:")
print(len(segment_summary))

print("\nSegment-duration summary:")
print(
    segment_summary["duration_minutes"].describe()
)

print(
    "\nDetected nominal burst:",
    nominal_burst_samples,
    "samples =",
    nominal_burst_samples / fs / 60.0,
    "minutes",
)

print(
    "Minimum accepted block:",
    minimum_block_samples,
    "samples",
)


# # ============================================================
# # CREATE ANALYSIS BLOCKS
# # ============================================================

analysis_blocks = []

for segment_id, segment in df.groupby(
    "segment_id",
    sort=True,
):
    segment = segment.reset_index(drop=True)

    blocks = split_segment_into_analysis_blocks(
        segment=segment,
        nominal_block_samples=nominal_burst_samples,
        minimum_block_samples=minimum_block_samples,
    )

    for block in blocks:
        block["source_segment_id"] = segment_id
        analysis_blocks.append(block)

print("\nAnalysis blocks produced:")
print(len(analysis_blocks))


# # ============================================================
# # COMPUTE SPECTRA FOR ALL CELLS
# # ============================================================

statistics_records = []
spectral_records = []

maximum_interpolation_gap_samples = max(
    1,
    int(
        round(
            maximum_interpolation_gap_seconds
            * fs
        )
    ),
)

for analysis_block_number, block in enumerate(
    analysis_blocks
):
    start_time = block["time"].iloc[0]
    end_time = block["time"].iloc[-1]

    source_segment_id = int(
        block["source_segment_id"].iloc[0]
    )

    subblock_number = int(
        block["subblock_number"].iloc[0]
    )

    sample_count = len(block)
    duration_minutes = sample_count / fs / 60.0

    for cell in cells:
        cell_number = cell["cell_number"]
        height_m = cell["height_relative_to_frame_bottom_m"]

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

        valid_fraction_before_interpolation = (
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
            "sample_count":
                sample_count,
            "duration_minutes":
                duration_minutes,
            "valid_fraction_before_interpolation":
                valid_fraction_before_interpolation,
        }

        rejection_reason = ""

        if sample_count < minimum_block_samples:
            rejection_reason = (
                "analysis block too short"
            )

        elif (
            valid_fraction_before_interpolation
            < minimum_valid_fraction
        ):
            rejection_reason = (
                "too many missing velocity samples"
            )

        if rejection_reason:
            statistics_records.append(
                {
                    **base_record,
                    "accepted": False,
                    "rejection_reason":
                        rejection_reason,
                    "valid_fraction_after_interpolation":
                        np.nan,
                    "mean_cross_current_m_s":
                        np.nan,
                    "mean_along_current_m_s":
                        np.nan,
                    "cross_background_change_m_s":
                        np.nan,
                    "along_background_change_m_s":
                        np.nan,
                    "cross_ig_variance_m2_s2":
                        np.nan,
                    "along_ig_variance_m2_s2":
                        np.nan,
                    "cross_ss_variance_m2_s2":
                        np.nan,
                    "along_ss_variance_m2_s2":
                        np.nan,
                    "cross_ig_rms_m_s":
                        np.nan,
                    "along_ig_rms_m_s":
                        np.nan,
                    "cross_ss_rms_m_s":
                        np.nan,
                    "along_ss_rms_m_s":
                        np.nan,
                    "total_horizontal_ig_variance_m2_s2":
                        np.nan,
                }
            )
            continue

        cross_series = interpolate_short_gaps(
            block[cross_column],
            limit_samples=maximum_interpolation_gap_samples,
        )

        along_series = interpolate_short_gaps(
            block[along_column],
            limit_samples=maximum_interpolation_gap_samples,
        )

        finite_after = (
            cross_series.notna()
            & along_series.notna()
        )

        valid_fraction_after_interpolation = (
            finite_after.mean()
        )

        if not finite_after.all():
            statistics_records.append(
                {
                    **base_record,
                    "accepted": False,
                    "rejection_reason":
                        "unfilled internal or edge velocity gaps",
                    "valid_fraction_after_interpolation":
                        valid_fraction_after_interpolation,
                    "mean_cross_current_m_s":
                        np.nan,
                    "mean_along_current_m_s":
                        np.nan,
                    "cross_background_change_m_s":
                        np.nan,
                    "along_background_change_m_s":
                        np.nan,
                    "cross_ig_variance_m2_s2":
                        np.nan,
                    "along_ig_variance_m2_s2":
                        np.nan,
                    "cross_ss_variance_m2_s2":
                        np.nan,
                    "along_ss_variance_m2_s2":
                        np.nan,
                    "cross_ig_rms_m_s":
                        np.nan,
                    "along_ig_rms_m_s":
                        np.nan,
                    "cross_ss_rms_m_s":
                        np.nan,
                    "along_ss_rms_m_s":
                        np.nan,
                    "total_horizontal_ig_variance_m2_s2":
                        np.nan,
                }
            )
            continue

        cross_raw = cross_series.to_numpy(
            dtype=np.float64,
        )

        along_raw = along_series.to_numpy(
            dtype=np.float64,
        )

        mean_cross_current = float(
            np.mean(cross_raw)
        )

        mean_along_current = float(
            np.mean(along_raw)
        )

        cross, cross_background = polynomial_detrend(
            cross_raw,
            order=detrend_order,
        )

        along, along_background = polynomial_detrend(
            along_raw,
            order=detrend_order,
        )

        cross_background_change = float(
            cross_background[-1]
            - cross_background[0]
        )

        along_background_change = float(
            along_background[-1]
            - along_background[0]
        )

        (
            frequency,
            cross_psd,
            along_psd,
        ) = calculate_autospectra(
            cross_shore=cross,
            alongshore=along,
            fs=fs,
            segment_seconds=welch_segment_seconds,
            overlap_fraction=overlap_fraction,
        )

        cross_ig_variance = integrate_spectral_band(
            frequency,
            cross_psd,
            ig_low_hz,
            ig_high_hz,
        )

        along_ig_variance = integrate_spectral_band(
            frequency,
            along_psd,
            ig_low_hz,
            ig_high_hz,
        )

        cross_ss_variance = integrate_spectral_band(
            frequency,
            cross_psd,
            ss_low_hz,
            min(ss_high_hz, 0.5 * fs),
        )

        along_ss_variance = integrate_spectral_band(
            frequency,
            along_psd,
            ss_low_hz,
            min(ss_high_hz, 0.5 * fs),
        )

        cross_ig_rms = (
            np.sqrt(cross_ig_variance)
            if np.isfinite(cross_ig_variance)
            else np.nan
        )

        along_ig_rms = (
            np.sqrt(along_ig_variance)
            if np.isfinite(along_ig_variance)
            else np.nan
        )

        cross_ss_rms = (
            np.sqrt(cross_ss_variance)
            if np.isfinite(cross_ss_variance)
            else np.nan
        )

        along_ss_rms = (
            np.sqrt(along_ss_variance)
            if np.isfinite(along_ss_variance)
            else np.nan
        )

        total_horizontal_ig_variance = (
            cross_ig_variance
            + along_ig_variance
        )

        statistics_records.append(
            {
                **base_record,
                "accepted": True,
                "rejection_reason": "",
                "valid_fraction_after_interpolation":
                    valid_fraction_after_interpolation,
                "mean_cross_current_m_s":
                    mean_cross_current,
                "mean_along_current_m_s":
                    mean_along_current,
                "cross_background_change_m_s":
                    cross_background_change,
                "along_background_change_m_s":
                    along_background_change,
                "cross_ig_variance_m2_s2":
                    cross_ig_variance,
                "along_ig_variance_m2_s2":
                    along_ig_variance,
                "cross_ss_variance_m2_s2":
                    cross_ss_variance,
                "along_ss_variance_m2_s2":
                    along_ss_variance,
                "cross_ig_rms_m_s":
                    cross_ig_rms,
                "along_ig_rms_m_s":
                    along_ig_rms,
                "cross_ss_rms_m_s":
                    cross_ss_rms,
                "along_ss_rms_m_s":
                    along_ss_rms,
                "total_horizontal_ig_variance_m2_s2":
                    total_horizontal_ig_variance,
            }
        )

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


# # ============================================================
# # SAVE RESULTS
# # ============================================================

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

# statistics_file = (
#     output_folder
#     / f"{case_id}_adcp_burst_spectral_statistics.csv"
# )

# spectra_file = (
#     output_folder
#     / f"{case_id}_adcp_velocity_spectra.pkl"
# )

# segment_file = (
#     output_folder
#     / f"{case_id}_adcp_segment_summary.csv"
# )

# statistics.to_csv(
#     statistics_file,
#     index=False,
# )

# spectra.to_pickle(
#     spectra_file,
# )

# segment_summary.to_csv(
#     segment_file,
#     index=False,
# )

# print("\nSaved segment summary:")
# print(segment_file)

# print("\nSaved spectral statistics:")
# print(statistics_file)

# print("\nSaved spectra:")
# print(spectra_file)

# print("\nAccepted cell/burst combinations:")
# print(statistics["accepted"].sum())

# print("Rejected cell/burst combinations:")
# print((~statistics["accepted"]).sum())

# print("\nRejection reasons:")
# print(
#     statistics["rejection_reason"]
#     .value_counts(dropna=False)
# )

# print("\nValid fraction before interpolation:")
# print(
#     statistics[
#         "valid_fraction_before_interpolation"
#     ].describe(
#         percentiles=[
#             0.01,
#             0.05,
#             0.25,
#             0.50,
#             0.75,
#             0.95,
#             0.99,
#         ]
#     )
# )

# print("\nValid fraction after interpolation:")
# print(
#     statistics[
#         "valid_fraction_after_interpolation"
#     ].describe(
#         percentiles=[
#             0.01,
#             0.05,
#             0.25,
#             0.50,
#             0.75,
#             0.95,
#             0.99,
#         ]
#     )
# )

# print("\nRejections by cell and reason:")
# print(
#     statistics.groupby(
#         [
#             "cell_number",
#             "rejection_reason",
#         ]
#     )
#     .size()
# )


# # ============================================================
# # ACCEPTED DATA
# # ============================================================

accepted = statistics.loc[
    statistics["accepted"]
].copy()

if accepted.empty:
    raise RuntimeError(
        "No ADCP cell/burst combinations passed quality control."
    )

accepted["mid_time"] = (
    accepted["start_time"]
    + (
        accepted["end_time"]
        - accepted["start_time"]
    ) / 2
)

accepted = accepted.sort_values(
    [
        "cell_number",
        "mid_time",
    ]
)

print("\nAccepted statistics by cell:")
print(
    accepted.groupby(
        [
            "cell_number",
            "height_relative_to_frame_bottom_m",
        ]
    )[
        [
            "cross_ig_rms_m_s",
            "along_ig_rms_m_s",
            "cross_ss_rms_m_s",
            "along_ss_rms_m_s",
        ]
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# ============================================================
# PLOTTING HELPERS: ONE SUBPLOT PER CELL
# ============================================================

def create_cell_subplot_grid(
    number_of_cells,
    figure_width=13,
    row_height=6.0,
):
    """
    Create one subplot for every selected ADCP-HR cell.
    """
    if number_of_cells <= 0:
        raise ValueError(
            "At least one plotting cell is required."
        )

    number_of_columns = min(
        2,
        number_of_cells,
    )

    number_of_rows = int(
        np.ceil(
            number_of_cells
            / number_of_columns
        )
    )

    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(
            figure_width,
            row_height * number_of_rows,
        ),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for axis in axes_flat[
        number_of_cells:
    ]:
        axis.set_visible(False)

    return fig, axes_flat

# Cells shown in the result figures.
selected_plot_cells = [
    1,
    5,
    9,
    13,
]

available_cell_numbers = {
    cell["cell_number"]
    for cell in cells
}

invalid_selected_cells = [
    cell_number
    for cell_number in selected_plot_cells
    if cell_number not in available_cell_numbers
]

if invalid_selected_cells:
    raise ValueError(
        "Selected plotting cells are not available: "
        f"{invalid_selected_cells}. "
        "Available cells are: "
        f"{sorted(available_cell_numbers)}"
    )

plot_cells = [
    cell
    for cell in cells
    if cell["cell_number"]
    in selected_plot_cells
]

# Preserve the order given in selected_plot_cells.
plot_cells = sorted(
    plot_cells,
    key=lambda cell:
        selected_plot_cells.index(
            cell["cell_number"]
        ),
)

print("\nCells selected for plotting:")

for cell in plot_cells:
    print(
        f"Cell {cell['cell_number']}: "
        f"z = "
        f"{cell['height_relative_to_frame_bottom_m']:.3f} m"
    )

# ============================================================
# TIME-VARYING IG RMS: 13 SUBPLOTS, BOTH COMPONENTS PER CELL
# ============================================================

fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
legend_handles = None
legend_labels = None

for axis, cell in zip(axes, plot_cells):
    cell_number = cell["cell_number"]
    height_m = cell["height_relative_to_frame_bottom_m"]

    cell_data = accepted.loc[
        accepted["cell_number"] == cell_number
    ].sort_values("mid_time").copy()

    axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

    if cell_data.empty:
        axis.text(0.5, 0.5, "No accepted bursts", ha="center", va="center", transform=axis.transAxes)
        continue

    dt_seconds = cell_data["mid_time"].diff().dt.total_seconds()
    cell_data["cross_plot"] = cell_data["cross_ig_rms_m_s"]
    cell_data["along_plot"] = cell_data["along_ig_rms_m_s"]

    large_gap = dt_seconds > plot_gap_seconds
    cell_data.loc[large_gap, ["cross_plot", "along_plot"]] = np.nan

    axis.plot(cell_data["mid_time"], cell_data["cross_plot"], label="Cross-shore")
    axis.plot(cell_data["mid_time"], cell_data["along_plot"], linewidth=0.5, label="Alongshore")
    axis.set_ylabel("IG RMS (m/s)")
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="x", rotation=30)

    if legend_handles is None:
        legend_handles, legend_labels = axis.get_legend_handles_labels()

for axis in axes[max(0, len(cells) - 3):len(cells)]:
    axis.set_xlabel("Time")

fig.suptitle(f"Time-varying IG velocity RMS ({case_label})", y=0.995)

if legend_handles:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.975),
    )

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ============================================================
# SELECT MOST ENERGETIC IG BURST ACROSS THE PROFILE
# ============================================================

burst_energy = (
    accepted.groupby("analysis_block_number", as_index=False)
    .agg(
        start_time=("start_time", "first"),
        end_time=("end_time", "first"),
        accepted_cell_count=("cell_number", "nunique"),
        profile_mean_total_ig_variance=("total_horizontal_ig_variance_m2_s2", "mean"),
        profile_median_total_ig_variance=("total_horizontal_ig_variance_m2_s2", "median"),
    )
)

minimum_cells_for_energetic_selection = max(2, int(np.ceil(0.75 * len(cells))))

energetic_candidates = burst_energy.loc[
    burst_energy["accepted_cell_count"] >= minimum_cells_for_energetic_selection
].copy()

if energetic_candidates.empty:
    energetic_candidates = burst_energy.copy()

most_energetic_row = energetic_candidates.loc[
    energetic_candidates["profile_mean_total_ig_variance"].idxmax()
]

most_energetic_block_number = int(most_energetic_row["analysis_block_number"])

most_energetic_spectra = spectra.loc[
    spectra["analysis_block_number"] == most_energetic_block_number
].copy()


# ============================================================
# MOST ENERGETIC BURST SPECTRA: 13 SUBPLOTS
# ============================================================

fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
legend_handles = None
legend_labels = None

for axis, cell in zip(axes, plot_cells):
    cell_number = cell["cell_number"]
    height_m = cell["height_relative_to_frame_bottom_m"]

    cell_spectrum = most_energetic_spectra.loc[
        (most_energetic_spectra["cell_number"] == cell_number)
        & (most_energetic_spectra["frequency_hz"] > 0)
    ].sort_values("frequency_hz")

    axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

    if cell_spectrum.empty:
        axis.text(0.5, 0.5, "No accepted spectrum", ha="center", va="center", transform=axis.transAxes)
        continue

    axis.loglog(
        cell_spectrum["frequency_hz"],
        cell_spectrum["cross_psd_m2_s2_hz"],
        label="Cross-shore",
    )
    axis.loglog(
        cell_spectrum["frequency_hz"],
        cell_spectrum["along_psd_m2_s2_hz"],
        label="Alongshore",
    )
    axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
    axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
    axis.grid(True, which="both", alpha=0.25)

    if legend_handles is None:
        legend_handles, legend_labels = axis.get_legend_handles_labels()

fig.suptitle(
    "Most energetic profile-wide IG burst\n"
    f"{case_label}: {most_energetic_row['start_time']} to {most_energetic_row['end_time']}",
    y=0.995,
)

if legend_handles:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.955),
    )

fig.tight_layout(rect=[0, 0, 1, 0.91])
plt.show()


# ============================================================
# MEDIAN SPECTRA: 13 SUBPLOTS
# ============================================================

median_spectra = (
    spectra.groupby(
        ["cell_number", "height_relative_to_frame_bottom_m", "frequency_hz"],
        as_index=False,
    )
    .agg(
        cross_psd_median=("cross_psd_m2_s2_hz", "median"),
        along_psd_median=("along_psd_m2_s2_hz", "median"),
        cross_psd_q25=("cross_psd_m2_s2_hz", lambda values: values.quantile(0.25)),
        cross_psd_q75=("cross_psd_m2_s2_hz", lambda values: values.quantile(0.75)),
        along_psd_q25=("along_psd_m2_s2_hz", lambda values: values.quantile(0.25)),
        along_psd_q75=("along_psd_m2_s2_hz", lambda values: values.quantile(0.75)),
    )
)

fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
legend_handles = None
legend_labels = None

for axis, cell in zip(axes, plot_cells):
    cell_number = cell["cell_number"]
    height_m = cell["height_relative_to_frame_bottom_m"]

    plot_data = median_spectra.loc[
        (median_spectra["cell_number"] == cell_number)
        & (median_spectra["frequency_hz"] > 0)
    ].sort_values("frequency_hz")

    axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

    if plot_data.empty:
        axis.text(0.5, 0.5, "No accepted spectra", ha="center", va="center", transform=axis.transAxes)
        continue

    axis.loglog(plot_data["frequency_hz"], plot_data["cross_psd_median"], label="Cross-shore median")
    axis.loglog(plot_data["frequency_hz"], plot_data["along_psd_median"], label="Alongshore median")
    axis.fill_between(
        plot_data["frequency_hz"],
        plot_data["cross_psd_q25"],
        plot_data["cross_psd_q75"],
        alpha=0.12,
    )
    axis.fill_between(
        plot_data["frequency_hz"],
        plot_data["along_psd_q25"],
        plot_data["along_psd_q75"],
        alpha=0.12,
    )
    axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
    axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
    axis.set_ylim(1e-4, 1e-1)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
    axis.grid(True, which="both", alpha=0.25)

    if legend_handles is None:
        legend_handles, legend_labels = axis.get_legend_handles_labels()

fig.suptitle(f"Median ADCP-HR velocity autospectra ({case_label})", y=0.995)

if legend_handles:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.965),
    )

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


# ============================================================
# MEAN SPECTRA: 13 SUBPLOTS
# ============================================================

mean_spectra = (
    spectra.groupby(
        ["cell_number", "height_relative_to_frame_bottom_m", "frequency_hz"],
        as_index=False,
    )
    .agg(
        cross_psd_mean=("cross_psd_m2_s2_hz", "mean"),
        along_psd_mean=("along_psd_m2_s2_hz", "mean"),
        cross_psd_std=("cross_psd_m2_s2_hz", "std"),
        along_psd_std=("along_psd_m2_s2_hz", "std"),
    )
)

fig, axes = create_cell_subplot_grid(len(plot_cells), figure_width=12, row_height=4.0)
legend_handles = None
legend_labels = None

for axis, cell in zip(axes, plot_cells):
    cell_number = cell["cell_number"]
    height_m = cell["height_relative_to_frame_bottom_m"]

    plot_data = mean_spectra.loc[
        (mean_spectra["cell_number"] == cell_number)
        & (mean_spectra["frequency_hz"] > 0)
    ].sort_values("frequency_hz")

    axis.set_title(f"Cell {cell_number}: z={height_m:.3f} m")

    if plot_data.empty:
        axis.text(0.5, 0.5, "No accepted spectra", ha="center", va="center", transform=axis.transAxes)
        continue

    axis.loglog(plot_data["frequency_hz"], plot_data["cross_psd_mean"], label="Cross-shore mean")
    axis.loglog(plot_data["frequency_hz"], plot_data["along_psd_mean"], label="Alongshore mean")
    axis.axvspan(ig_low_hz, ig_high_hz, alpha=0.20, label="IG band")
    axis.axvspan(ss_low_hz, min(ss_high_hz, 0.5 * fs), alpha=0.10, label="Sea-swell band")
    axis.set_ylim(2e-4, 8e-2)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
    axis.grid(True, which="both", alpha=0.25)

    if legend_handles is None:
        legend_handles, legend_labels = axis.get_legend_handles_labels()

fig.suptitle(f"Mean ADCP-HR velocity autospectra ({case_label})", y=0.995)

if legend_handles:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.965),
    )

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

