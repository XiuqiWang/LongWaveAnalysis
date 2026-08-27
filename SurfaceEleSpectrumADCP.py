# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:10:21 2026

@author: WangX3
"""

# -*- coding: utf-8 -*-
"""
Compute pressure-derived surface-elevation spectra and spectral
significant wave heights from processed ADCP pressure Parquet data.

Workflow
--------
1. Read corrected ADCP pressure and sensor metadata.
2. Detect the sampling frequency.
3. Detect natural continuous recording segments.
4. Detect nominal burst length and split long segments.
5. Convert corrected pressure to pressure head.
6. Estimate mean water depth.
7. Detrend each burst.
8. Calculate Welch pressure-head autospectrum.
9. Correct pressure attenuation using linear wave theory.
10. Reconstruct surface-elevation spectrum.
11. Calculate IG and sea-swell variance and Hm0.
12. Save burst statistics and update shared cross-case wave-height CSV.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# USER SETTINGS
# ============================================================

input_file = Path(
    r"C:\dev\Python\LongWaveAnalysis\Processed"
    r"\Pressure_DVN_F3_ADCP.parquet"
)

case_id = "DVN_F3_ADCP"
case_label = "DVN F3 ADCP"

pressure_column = "pressure_corrected"
z_pressure_m = 2.293 #ADCP pressure sensor
pressure_units = "Pa"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)
output_folder.mkdir(parents=True, exist_ok=True)

rho_water_kg_m3 = 1025.0
gravity_m_s2 = 9.81

# Welch settings
welch_segment_seconds = 512.0
overlap_fraction = 0.5

# Frequency bands
ig_low_hz = 0.005
ig_high_hz = 0.05

ss_low_hz = 0.05
ss_high_hz = 1.0

# Common upper SS cutoff for comparison between F1/F3/etc.
common_ss_high_hz = 0.185

# QC
minimum_valid_fraction = 0.98
minimum_block_fraction = 0.95
minimum_water_depth_m = 0.25

# Approximate deployment depth; change for frame
expected_water_depth_m = 12.0       # F3 ~12 m
maximum_depth_deviation_m = 3.0

detrend_order = 2

# Maximum allowed pressure-to-surface amplitude correction 1/Kp.
max_pressure_amplitude_gain = 10.0

if not (ss_low_hz < common_ss_high_hz <= ss_high_hz):
    raise ValueError(
        "common_ss_high_hz must be greater than ss_low_hz "
        "and no larger than ss_high_hz."
    )


# ============================================================
# FUNCTIONS
# ============================================================

def identify_continuous_segments(df, gap_threshold_seconds=1.0):
    output = df.copy()
    dt = output["time"].diff().dt.total_seconds()

    new_segment = (
        dt.isna()
        | (dt > gap_threshold_seconds)
        | (dt <= 0)
    )

    output["segment_id"] = new_segment.cumsum()
    return output


def split_segment_into_analysis_blocks(
    segment,
    nominal_block_samples,
    minimum_block_samples,
):
    blocks = []
    start = 0
    subblock_number = 0
    n = len(segment)

    while n - start >= nominal_block_samples:
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
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite values.")

    x = np.linspace(-1.0, 1.0, len(values))
    coefficients = np.polyfit(x, values, deg=order)
    trend = np.polyval(coefficients, x)

    return values - trend, trend


def calculate_autospectrum(
    values,
    fs,
    segment_seconds=512.0,
    overlap_fraction=0.5,
):
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite values.")

    nperseg = min(
        int(round(segment_seconds * fs)),
        len(values),
    )

    if nperseg < 16:
        raise ValueError(
            "Analysis block is too short for Welch analysis."
        )

    noverlap = min(
        int(round(overlap_fraction * nperseg)),
        nperseg - 1,
    )

    return welch(
        values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )


def detect_sampling_frequency(time):
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
            "Could not estimate sampling interval."
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
            "Detected nonpositive sampling interval."
        )

    return 1.0 / median_dt, median_dt


def solve_wavenumber(
    frequency_hz,
    water_depth_m,
    gravity=9.81,
    tolerance=1e-12,
    maximum_iterations=100,
):
    """Solve omega^2 = g k tanh(kh)."""

    frequency = np.asarray(
        frequency_hz,
        dtype=np.float64,
    )

    omega = 2.0 * np.pi * frequency

    shallow_guess = (
        omega
        / np.sqrt(
            gravity * water_depth_m
        )
    )

    deep_guess = (
        omega**2
        / gravity
    )

    k = np.maximum(
        shallow_guess,
        deep_guess,
    )

    k[omega == 0] = 0.0

    for _ in range(maximum_iterations):
        kh = k * water_depth_m
        tanh_kh = np.tanh(kh)

        function = (
            gravity * k * tanh_kh
            - omega**2
        )

        sech_squared_kh = np.maximum(
            1.0 - tanh_kh**2,
            0.0,
        )

        derivative = gravity * (
            tanh_kh
            + kh * sech_squared_kh
        )

        update = np.zeros_like(k)

        valid = (
            (omega > 0)
            & np.isfinite(derivative)
            & (derivative != 0)
        )

        update[valid] = (
            function[valid]
            / derivative[valid]
        )

        new_k = np.maximum(
            k - update,
            0.0,
        )

        if np.nanmax(
            np.abs(new_k - k)
        ) < tolerance:
            k = new_k
            break

        k = new_k

    return k


def pressure_response_factor(
    frequency_hz,
    water_depth_m,
    sensor_height_above_bed_m,
    gravity=9.81,
):
    """
    Linear pressure response:

        Kp = cosh(k z_sensor) / cosh(k h)
    """

    if water_depth_m <= sensor_height_above_bed_m:
        raise ValueError(
            "Water depth must exceed sensor height."
        )

    frequency = np.asarray(
        frequency_hz,
        dtype=np.float64,
    )

    k = solve_wavenumber(
        frequency,
        water_depth_m,
        gravity,
    )

    kz = (
        k
        * sensor_height_above_bed_m
    )

    kh = (
        k
        * water_depth_m
    )

    # Numerically stable cosh(kz)/cosh(kh)
    kp = (
        np.exp(kz - kh)
        * (
            1.0
            + np.exp(-2.0 * kz)
        )
        / (
            1.0
            + np.exp(-2.0 * kh)
        )
    )

    kp[frequency == 0] = 1.0

    return kp, k


def pressure_to_surface_spectrum(
    frequency,
    pressure_head_psd,
    water_depth_m,
    sensor_height_above_bed_m,
    gravity=9.81,
    maximum_amplitude_gain=10.0,
):
    """
    Convert pressure-head PSD to surface-elevation PSD:

        S_eta = S_pressure_head / Kp^2
    """

    kp, wavenumber = (
        pressure_response_factor(
            frequency,
            water_depth_m,
            sensor_height_above_bed_m,
            gravity,
        )
    )

    minimum_kp = (
        1.0
        / maximum_amplitude_gain
    )

    valid_kp = (
        np.isfinite(kp)
        & (kp >= minimum_kp)
    )

    amplitude_gain = np.full_like(
        kp,
        np.inf,
        dtype=np.float64,
    )

    np.divide(
        1.0,
        kp,
        out=amplitude_gain,
        where=valid_kp,
    )

    reliable = (
        valid_kp
        & np.isfinite(amplitude_gain)
        & (
            amplitude_gain
            <= maximum_amplitude_gain
        )
    )

    eta_psd = np.full_like(
        pressure_head_psd,
        np.nan,
    )

    eta_psd[reliable] = (
        pressure_head_psd[reliable]
        / kp[reliable] ** 2
    )

    return (
        eta_psd,
        kp,
        amplitude_gain,
        reliable,
        wavenumber,
    )


def contiguous_reliable_upper_frequency(
    frequency,
    reliable,
    lower_hz,
    requested_upper_hz,
):
    frequency = np.asarray(
        frequency,
        dtype=float,
    )

    reliable = np.asarray(
        reliable,
        dtype=bool,
    )

    indices = np.flatnonzero(
        (frequency >= lower_hz)
        & (frequency < requested_upper_hz)
    )

    accepted_indices = []

    for index in indices:
        if reliable[index]:
            accepted_indices.append(index)
        else:
            break

    if len(accepted_indices) < 2:
        return np.nan

    df = np.nanmedian(
        np.diff(frequency)
    )

    return min(
        requested_upper_hz,
        frequency[
            accepted_indices[-1]
        ] + df,
    )


def integrate_spectral_band(
    frequency,
    psd,
    lower_hz,
    upper_hz,
):
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

    selected_frequency = (
        frequency[selected]
    )

    selected_psd = (
        psd[selected]
    )

    df = np.nanmedian(
        np.diff(frequency)
    )

    # Do not integrate across unreliable spectral gaps.
    if np.any(
        np.diff(selected_frequency)
        > 1.5 * df
    ):
        return np.nan

    return np.trapezoid(
        selected_psd,
        selected_frequency,
    )


def pressure_to_pa(values, units):
    """
    Convert corrected pressure to Pa.

    Handles common units stored in pressure files.
    """

    units = str(units).strip().lower()
    values = np.asarray(values, dtype=np.float64)

    if units in {
        "pa",
        "pascal",
        "pascals",
    }:
        factor = 1.0

    elif units in {
        "kpa",
        "kilopascal",
        "kilopascals",
    }:
        factor = 1000.0

    elif units in {
        "hpa",
        "mbar",
        "millibar",
    }:
        factor = 100.0

    elif units in {
        "bar",
        "bars",
    }:
        factor = 1e5

    elif units in {
        "dbar",
        "decibar",
        "decibars",
    }:
        factor = 1e4

    else:
        raise ValueError(
            "Unrecognized pressure units "
            f"{units!r}. Check the metadata CSV."
        )

    return values * factor

# ============================================================
# READ PRESSURE PARQUET
# ============================================================

print("\nReading:")
print(input_file)

df = pd.read_parquet(
    input_file
)

required_columns = {
    "time",
    pressure_column,
}

missing = (
    required_columns
    - set(df.columns)
)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

df = df[
    [
        "time",
        pressure_column,
    ]
].copy()

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
    errors="coerce",
)

df[pressure_column] = pd.to_numeric(
    df[pressure_column],
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

print(
    "Start:",
    df["time"].iloc[0],
)

print(
    "End:  ",
    df["time"].iloc[-1],
)

print(
    "Rows: ",
    len(df),
)

print(
    "Missing pressure:",
    df[pressure_column].isna().sum(),
)


# ============================================================
# DETECT NATIVE SAMPLING FREQUENCY
# ============================================================

fs, regular_dt_seconds = (
    detect_sampling_frequency(
        df["time"]
    )
)

print("\nDetected ADCP pressure sampling:")
print("dt =", regular_dt_seconds, "s")
print("fs =", fs, "Hz")


# ============================================================
# DETECT NATURAL BURSTS
# ============================================================
fs, regular_dt_seconds = detect_sampling_frequency(df["time"])

gap_threshold_seconds = max(
    3.0 * regular_dt_seconds,
    2.0,
)

print("\nDetected sampling:")
print("dt =", regular_dt_seconds, "s")
print("fs =", fs, "Hz")
print("Gap threshold =", gap_threshold_seconds, "s")

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
    segment_summary["sample_count"]
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
    "\nContinuous segments:",
    len(segment_summary),
)

print(
    "Detected nominal burst:",
    nominal_burst_samples,
    "samples =",
    nominal_burst_samples / fs / 60.0,
    "minutes",
)


# ============================================================
# CREATE ANALYSIS BLOCKS
# ============================================================

analysis_blocks = []

for segment_id, segment in df.groupby(
    "segment_id",
    sort=True,
):
    blocks = split_segment_into_analysis_blocks(
        segment.reset_index(drop=True),
        nominal_burst_samples,
        minimum_block_samples,
    )

    for block in blocks:
        block[
            "source_segment_id"
        ] = segment_id

        analysis_blocks.append(
            block
        )

print(
    "Analysis blocks:",
    len(analysis_blocks),
)


# ============================================================
# COMPUTE SURFACE-ELEVATION SPECTRA + Hm0
# ============================================================

statistics_records = []
spectral_records = []
directional_pressure_records = []

for analysis_block_number, block in enumerate(
    analysis_blocks
):
    start_time = block["time"].iloc[0]
    end_time = block["time"].iloc[-1]
    mid_time = (
        start_time
        + (end_time - start_time) / 2
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
    duration_minutes = sample_count / fs / 60.0

    valid_fraction = (
        block[pressure_column]
        .notna()
        .mean()
    )

    base_record = {
        "analysis_block_number":
            analysis_block_number,
        "source_segment_id":
            source_segment_id,
        "subblock_number":
            subblock_number,
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
        "valid_fraction":
            valid_fraction,
    }

    def rejected_record(
        reason,
        mean_pressure_pa=np.nan,
        mean_pressure_head_m=np.nan,
        mean_water_depth_m=np.nan,
    ):
        return {
            **base_record,
            "accepted": False,
            "rejection_reason": reason,
            "mean_pressure_pa":
                mean_pressure_pa,
            "mean_pressure_head_m":
                mean_pressure_head_m,
            "mean_water_depth_m":
                mean_water_depth_m,
            "ig_variance_m2":
                np.nan,
            "ss_variance_m2":
                np.nan,
            "ss_common_fc_variance_m2":
                np.nan,
            "hm0_ig_m":
                np.nan,
            "hm0_ss_m":
                np.nan,
            "hm0_ss_common_fc_m":
                np.nan,
            "ig_to_ss_variance_ratio":
                np.nan,
            "ig_to_ss_common_fc_variance_ratio":
                np.nan,
            "ig_effective_high_hz":
                np.nan,
            "ss_effective_high_hz":
                np.nan,
            "common_fc_reliable":
                False,
            "max_reliable_frequency_hz":
                np.nan,
        }


    # --------------------------------------------------------
    # Block QC
    # --------------------------------------------------------

    if sample_count < minimum_block_samples:
        statistics_records.append(
            rejected_record(
                "analysis block too short"
            )
        )
        continue

    if valid_fraction < minimum_valid_fraction:
        statistics_records.append(
            rejected_record(
                "too many missing pressure samples"
            )
        )
        continue


    # --------------------------------------------------------
    # Fill isolated missing values
    # --------------------------------------------------------

    pressure_native = (
        block[pressure_column]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        pressure_native
    ).all():
        statistics_records.append(
            rejected_record(
                "non-finite pressure after interpolation"
            )
        )
        continue


    # --------------------------------------------------------
    # Convert pressure units to Pa
    # --------------------------------------------------------

    pressure_pa = pressure_to_pa(
        pressure_native,
        pressure_units,
    )

    mean_pressure_pa = float(
        np.mean(
            pressure_pa
        )
    )

    pressure_std_pa = float(
        np.std(
            pressure_pa
        )
    )

    if (
        not np.isfinite(pressure_std_pa)
        or pressure_std_pa <= 0
    ):
        statistics_records.append(
            rejected_record(
                "frozen or constant pressure",
                mean_pressure_pa=
                    mean_pressure_pa,
            )
        )
        continue


    # --------------------------------------------------------
    # Pressure head + water depth
    # --------------------------------------------------------

    pressure_head_m = (
        pressure_pa
        / (
            rho_water_kg_m3
            * gravity_m_s2
        )
    )

    mean_pressure_head_m = float(
        np.mean(
            pressure_head_m
        )
    )

    # Hydrostatic relation:
    #
    # P/(rho g) = h - z_sensor
    #
    # therefore
    #
    # h = P/(rho g) + z_sensor

    mean_water_depth_m = (
        mean_pressure_head_m
        + z_pressure_m
    )

    if (
        not np.isfinite(
            mean_water_depth_m
        )
        or (
            mean_water_depth_m
            < minimum_water_depth_m
        )
        or (
            mean_water_depth_m
            <= z_pressure_m
        )
    ):
        statistics_records.append(
            rejected_record(
                "invalid estimated mean water depth",
                mean_pressure_pa,
                mean_pressure_head_m,
                mean_water_depth_m,
            )
        )
        continue

    if (
        abs(
            mean_water_depth_m
            - expected_water_depth_m
        )
        > maximum_depth_deviation_m
    ):
        statistics_records.append(
            rejected_record(
                "implausible mean water depth",
                mean_pressure_pa,
                mean_pressure_head_m,
                mean_water_depth_m,
            )
        )
        continue


    # --------------------------------------------------------
    # Detrend pressure head
    # --------------------------------------------------------

    (
        pressure_head_anomaly,
        pressure_head_background,
    ) = polynomial_detrend(
        pressure_head_m,
        order=detrend_order,
    )


    # --------------------------------------------------------
    # Pressure-head PSD
    # --------------------------------------------------------

    (
        frequency,
        pressure_head_psd,
    ) = calculate_autospectrum(
        pressure_head_anomaly,
        fs,
        welch_segment_seconds,
        overlap_fraction,
    )


    # --------------------------------------------------------
    # Surface-elevation PSD
    # --------------------------------------------------------

    (
        eta_psd,
        kp,
        amplitude_gain,
        reliable,
        wavenumber,
    ) = pressure_to_surface_spectrum(
        frequency,
        pressure_head_psd,
        mean_water_depth_m,
        z_pressure_m,
        gravity_m_s2,
        max_pressure_amplitude_gain,
    )


    # --------------------------------------------------------
    # Reliable frequency limits
    # --------------------------------------------------------

    ig_effective_high_hz = (
        contiguous_reliable_upper_frequency(
            frequency,
            reliable,
            ig_low_hz,
            ig_high_hz,
        )
    )

    ss_effective_high_hz = (
        contiguous_reliable_upper_frequency(
            frequency,
            reliable,
            ss_low_hz,
            ss_high_hz,
        )
    )

    reliable_positive = frequency[
        reliable
        & (frequency > 0)
    ]

    max_reliable_frequency_hz = (
        float(
            reliable_positive.max()
        )
        if len(reliable_positive)
        else np.nan
    )


    # --------------------------------------------------------
    # IG variance
    # --------------------------------------------------------

    ig_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ig_low_hz,
            min(
                ig_high_hz,
                ig_effective_high_hz,
            ),
        )
        if np.isfinite(
            ig_effective_high_hz
        )
        else np.nan
    )


    # --------------------------------------------------------
    # Recoverable SS variance
    # --------------------------------------------------------

    ss_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ss_low_hz,
            min(
                ss_high_hz,
                ss_effective_high_hz,
            ),
        )
        if np.isfinite(
            ss_effective_high_hz
        )
        else np.nan
    )


    # --------------------------------------------------------
    # Fixed common-frequency SS variance
    # --------------------------------------------------------

    common_fc_reliable = (
        np.isfinite(
            ss_effective_high_hz
        )
        and (
            ss_effective_high_hz
            >= common_ss_high_hz
        )
    )

    ss_common_fc_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ss_low_hz,
            common_ss_high_hz,
        )
        if common_fc_reliable
        else np.nan
    )


    # --------------------------------------------------------
    # Spectral QC
    # --------------------------------------------------------

    if (
        not np.isfinite(
            ig_variance_m2
        )
        or ig_variance_m2 <= 0
    ):
        statistics_records.append(
            rejected_record(
                "invalid or nonpositive IG spectral variance",
                mean_pressure_pa,
                mean_pressure_head_m,
                mean_water_depth_m,
            )
        )
        continue
    
    # ============================================================
    # SAVE TIME-RESOLVED PRESSURE FOR LATER
    # PRESSURE-VELOCITY DIRECTIONAL ANALYSIS
    # ============================================================
    
    directional_pressure_records.append(
        pd.DataFrame({
            "case_id": case_id,
            "case_name": case_label,
            "analysis_block_number": analysis_block_number,
            "source_segment_id": source_segment_id,
            "subblock_number": subblock_number,
            "start_time": start_time,
            "end_time": end_time,
            "mid_time": mid_time,
            "time": block["time"].to_numpy(),
            "sensor_height_above_bed_m": z_pressure_m,
            "mean_water_depth_m": mean_water_depth_m,
            "pressure_corrected_pa": pressure_pa,
            "pressure_head_m": pressure_head_m,
            "pressure_head_detrended_m": pressure_head_anomaly,
        })
    )


    # --------------------------------------------------------
    # Hm0
    # --------------------------------------------------------

    hm0_ig_m = (
        4.0
        * np.sqrt(
            ig_variance_m2
        )
    )

    hm0_ss_m = (
        4.0
        * np.sqrt(
            ss_variance_m2
        )
        if (
            np.isfinite(
                ss_variance_m2
            )
            and ss_variance_m2 > 0
        )
        else np.nan
    )

    hm0_ss_common_fc_m = (
        4.0
        * np.sqrt(
            ss_common_fc_variance_m2
        )
        if (
            np.isfinite(
                ss_common_fc_variance_m2
            )
            and ss_common_fc_variance_m2 > 0
        )
        else np.nan
    )


    # --------------------------------------------------------
    # IG / SS variance ratios
    # --------------------------------------------------------

    ig_to_ss_variance_ratio = (
        ig_variance_m2
        / ss_variance_m2
        if (
            np.isfinite(
                ss_variance_m2
            )
            and ss_variance_m2 > 0
        )
        else np.nan
    )

    ig_to_ss_common_fc_variance_ratio = (
        ig_variance_m2
        / ss_common_fc_variance_m2
        if (
            np.isfinite(
                ss_common_fc_variance_m2
            )
            and ss_common_fc_variance_m2 > 0
        )
        else np.nan
    )


    # --------------------------------------------------------
    # Save accepted block
    # --------------------------------------------------------

    statistics_records.append({
        **base_record,
        "accepted": True,
        "rejection_reason": "",
        "mean_pressure_pa":
            mean_pressure_pa,
        "mean_pressure_head_m":
            mean_pressure_head_m,
        "mean_water_depth_m":
            mean_water_depth_m,
        "ig_variance_m2":
            ig_variance_m2,
        "ss_variance_m2":
            ss_variance_m2,
        "ss_common_fc_variance_m2":
            ss_common_fc_variance_m2,
        "hm0_ig_m":
            hm0_ig_m,
        "hm0_ss_m":
            hm0_ss_m,
        "hm0_ss_common_fc_m":
            hm0_ss_common_fc_m,
        "common_ss_high_hz":
            common_ss_high_hz,
        "common_fc_reliable":
            common_fc_reliable,
        "ig_to_ss_variance_ratio":
            ig_to_ss_variance_ratio,
        "ig_to_ss_common_fc_variance_ratio":
            ig_to_ss_common_fc_variance_ratio,
        "ig_effective_high_hz":
            ig_effective_high_hz,
        "ss_effective_high_hz":
            ss_effective_high_hz,
        "max_reliable_frequency_hz":
            max_reliable_frequency_hz,
    })


    # --------------------------------------------------------
    # Save spectrum
    # --------------------------------------------------------

    spectral_records.append(
        pd.DataFrame({
            "analysis_block_number":
                analysis_block_number,
            "source_segment_id":
                source_segment_id,
            "subblock_number":
                subblock_number,
            "start_time":
                start_time,
            "end_time":
                end_time,
            "mid_time":
                mid_time,
            "mean_water_depth_m":
                mean_water_depth_m,
            "frequency_hz":
                frequency,
            "pressure_head_psd_m2_hz":
                pressure_head_psd,
            "eta_psd_m2_hz":
                eta_psd,
            "pressure_response_factor":
                kp,
            "pressure_amplitude_gain":
                amplitude_gain,
            "pressure_correction_reliable":
                reliable,
            "wavenumber_rad_m":
                wavenumber,
        })
    )

# ============================================================
# BUILD TIME-RESOLVED PRESSURE TABLE
# ============================================================

directional_pressure = (
    pd.concat(
        directional_pressure_records,
        ignore_index=True,
    )
    if directional_pressure_records
    else pd.DataFrame()
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
    .sort_values("mid_time")
)

if accepted.empty:
    raise RuntimeError(
        "No pressure blocks passed QC."
    )

# ============================================================
# SAVE PRESSURE TIME SERIES FOR DIRECTIONAL ANALYSIS
# SHARED FILE FOR F1 AND F3
# ============================================================

directional_pressure_file = (
    output_folder
    / "all_cases_ADCP_directional_pressure.parquet"
)

if directional_pressure.empty:
    print(
        "\nNo accepted pressure time series "
        "available for directional analysis."
    )

else:
    if directional_pressure_file.exists():
        old = pd.read_parquet(
            directional_pressure_file
        )

        if "case_id" not in old.columns:
            raise KeyError(
                "Existing directional-pressure file "
                "does not contain case_id."
            )

        # Replace this case when rerunning it.
        old = old.loc[
            old["case_id"] != case_id
        ]

        directional_pressure = pd.concat(
            [old, directional_pressure],
            ignore_index=True,
        )

    directional_pressure = (
        directional_pressure
        .sort_values(
            [
                "case_id",
                "time",
                "analysis_block_number",
            ]
        )
        .reset_index(drop=True)
    )

    directional_pressure.to_parquet(
        directional_pressure_file,
        index=False,
    )

    print(
        "\nSaved directional pressure time series:\n"
        f"{directional_pressure_file}"
    )

    print(
        directional_pressure
        .groupby("case_id")
        .size()
    )

# ============================================================
# SAVE CASE STATISTICS
# ============================================================

statistics_file = (
    output_folder
    / f"{case_id}_pressure_burst_statistics.csv"
)

spectra_file = (
    output_folder
    / f"{case_id}_surface_elevation_spectra.pkl"
)

statistics.to_csv(
    statistics_file,
    index=False,
)

spectra.to_pickle(
    spectra_file
)


# ============================================================
# SAVE BURST-LEVEL PRESSURE / WAVE STATISTICS
# SHARED FILE FOR F1 AND F3
# ============================================================

pressure_statistics_file = (
    output_folder
    / "all_cases_ADCP_pressure_wave_statistics.csv"
)

pressure_save_columns = [
    "analysis_block_number",
    "source_segment_id",
    "subblock_number",
    "start_time",
    "end_time",
    "mid_time",
    "sample_count",
    "duration_minutes",
    "valid_fraction",
    "mean_pressure_pa",
    "mean_pressure_head_m",
    "mean_water_depth_m",
    "ig_variance_m2",
    "ss_variance_m2",
    "ss_common_fc_variance_m2",
    "hm0_ig_m",
    "hm0_ss_m",
    "hm0_ss_common_fc_m",
    "ig_to_ss_variance_ratio",
    "ig_to_ss_common_fc_variance_ratio",
    "ig_effective_high_hz",
    "ss_effective_high_hz",
    "common_fc_reliable",
    "max_reliable_frequency_hz",
]

pressure_save = accepted[
    pressure_save_columns
].copy()

pressure_save.insert(
    0,
    "case_name",
    case_label,
)

pressure_save.insert(
    0,
    "case_id",
    case_id,
)

# Save processing parameters with every burst
pressure_save["sensor_height_above_bed_m"] = z_pressure_m
pressure_save["sampling_frequency_hz"] = fs
pressure_save["ig_low_hz"] = ig_low_hz
pressure_save["ig_high_hz"] = ig_high_hz
pressure_save["ss_low_hz"] = ss_low_hz
pressure_save["ss_high_hz"] = ss_high_hz
pressure_save["common_ss_high_hz"] = common_ss_high_hz
pressure_save["detrend_order"] = detrend_order
pressure_save["welch_segment_seconds"] = welch_segment_seconds
pressure_save["overlap_fraction"] = overlap_fraction
pressure_save["max_pressure_amplitude_gain"] = (
    max_pressure_amplitude_gain
)

if pressure_statistics_file.exists():
    old = pd.read_csv(
        pressure_statistics_file,
        parse_dates=[
            "start_time",
            "end_time",
            "mid_time",
        ],
    )

    if "case_id" not in old.columns:
        raise KeyError(
            "Existing pressure-statistics file "
            "does not contain case_id."
        )

    # Replace current case rather than duplicate it
    old = old.loc[
        old["case_id"] != case_id
    ]

    pressure_save = pd.concat(
        [old, pressure_save],
        ignore_index=True,
    )

pressure_save = (
    pressure_save
    .sort_values(
        [
            "case_id",
            "mid_time",
            "analysis_block_number",
        ]
    )
    .reset_index(drop=True)
)

pressure_save.to_csv(
    pressure_statistics_file,
    index=False,
)

# ============================================================
# SUMMARY
# ============================================================

print(
    "\nAccepted blocks:",
    statistics["accepted"].sum(),
)

print(
    "Rejected blocks:",
    (~statistics["accepted"]).sum(),
)

print(
    "\nAccepted-burst summary:"
)

print(
    accepted[
        [
            "mean_water_depth_m",
            "hm0_ig_m",
            "hm0_ss_m",
            "hm0_ss_common_fc_m",
            "ig_to_ss_variance_ratio",
            "ss_effective_high_hz",
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
# TIME SERIES OF Hm0
# ============================================================

plot_data = accepted.copy()

dt = (
    plot_data["mid_time"]
    .diff()
    .dt.total_seconds()
)

large_gap = (
    dt > 3600
)

plot_data.loc[
    large_gap,
    [
        "hm0_ig_m",
        "hm0_ss_m",
        "hm0_ss_common_fc_m",
    ],
] = np.nan

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    plot_data["mid_time"],
    plot_data["hm0_ig_m"],
    label=r"$H_{m0,\mathrm{IG}}$",
)

plt.plot(
    plot_data["mid_time"],
    plot_data["hm0_ss_m"],
    label=r"$H_{m0,\mathrm{SS}}$ (automatic cutoff)",
)

plt.plot(
    plot_data["mid_time"],
    plot_data["hm0_ss_common_fc_m"],
    label=(
        rf"$H_{{m0,\mathrm{{SS}}}}$ "
        rf"($f_c={common_ss_high_hz:.3f}$ Hz)"
    ),
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Spectral significant wave height (m)"
)

plt.title(
    case_label
)

plt.legend()
plt.grid(
    True,
    alpha=0.25,
)
plt.tight_layout()
plt.show()


# ============================================================
# MOST ENERGETIC IG SPECTRUM
# ============================================================

energetic = (
    accepted.dropna(
        subset=["ig_variance_m2"]
    )
)

if (
    not energetic.empty
    and not spectra.empty
):
    row = energetic.loc[
        energetic[
            "ig_variance_m2"
        ].idxmax()
    ]

    block_number = int(
        row[
            "analysis_block_number"
        ]
    )

    selected = spectra.loc[
        spectra[
            "analysis_block_number"
        ]
        == block_number
    ]

    p = selected.loc[
        (
            selected[
                "frequency_hz"
            ] > 0
        )
        & selected[
            "eta_psd_m2_hz"
        ].notna()
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.loglog(
        p[
            "frequency_hz"
        ],
        p[
            "eta_psd_m2_hz"
        ],
        label="Surface elevation",
    )

    plt.axvspan(
        ig_low_hz,
        ig_high_hz,
        alpha=0.20,
        label="IG band",
    )

    plt.axvspan(
        ss_low_hz,
        ss_high_hz,
        alpha=0.10,
        label="Sea-swell band",
    )

    cutoff = row[
        "ss_effective_high_hz"
    ]

    if np.isfinite(cutoff):
        plt.axvline(
            cutoff,
            linestyle="--",
            label=(
                f"Reliable cutoff "
                f"({cutoff:.3f} Hz)"
            ),
        )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        r"$S_{\eta\eta}$ (m$^2$/Hz)"
    )

    plt.title(
        "Most energetic IG burst\n"
        f"{case_label}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()