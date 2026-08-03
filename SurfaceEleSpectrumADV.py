# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:08:28 2026

@author: WangX3

Compute pressure-derived surface-elevation autospectra and band-limited
spectral wave heights from a processed KG2 P_APC Parquet file.

The workflow mirrors the velocity script:
1. detect natural continuous bursts from timestamp gaps;
2. detect the nominal burst length;
3. split long segments into nominal analysis blocks;
4. convert atmospheric-pressure-corrected pressure to pressure head;
5. estimate mean water depth;
6. detrend each block;
7. calculate a Welch autospectrum;
8. correct for depth attenuation using linear wave theory;
9. calculate Hm0,IG and pressure-recoverable Hm0,SS.

Important: near-bed pressure cannot reliably recover arbitrarily high
sea-swell frequencies. Frequencies needing an amplitude correction greater
than max_pressure_amplitude_gain are excluded, and the effective upper
sea-swell frequency is saved for every block.
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
    r"\Pressure_F1_ADV01_APC.parquet"
)

case_id = "DVN_F1_ADV01"
case_label = "DVN F1 ADV01"

pressure_column = "pressure_apc_ADV01_90cm_pa"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)
output_folder.mkdir(parents=True, exist_ok=True)

fs = 16.0
z_pressure_m = 0.89 # Change for Frame!

rho_water_kg_m3 = 1025.0
gravity_m_s2 = 9.81

gap_threshold_seconds = 1.0

welch_segment_seconds = 512.0
overlap_fraction = 0.5

ig_low_hz = 0.005
ig_high_hz = 0.05

ss_low_hz = 0.05
ss_high_hz = 1.0

# Common upper sea-swell cutoff used for cross-frame comparisons.
# Keep this identical in every case script.
common_ss_high_hz = 0.185

minimum_valid_fraction = 0.98
minimum_block_fraction = 0.95
minimum_water_depth_m = 0.25
expected_water_depth_m = 20 # Change for Frame!
maximum_depth_deviation_m = 3.0

detrend_order = 2

# Maximum permitted free-surface amplitude gain, 1/Kp.
max_pressure_amplitude_gain = 10.0

if not (ss_low_hz < common_ss_high_hz <= ss_high_hz):
    raise ValueError(
        "common_ss_high_hz must be greater than ss_low_hz and "
        "no larger than ss_high_hz."
    )


# ============================================================
# FUNCTIONS
# ============================================================

def identify_continuous_segments(df, gap_threshold_seconds=1.0):
    """Start a new segment at a large or nonpositive timestamp step."""
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
    """Split one continuous segment into nominal-length blocks."""
    blocks = []
    number_of_samples = len(segment)
    start = 0
    subblock_number = 0

    while number_of_samples - start >= nominal_block_samples:
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
    """Remove a polynomial background from one pressure-head block."""
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
    """Calculate a one-sided Welch PSD for an already detrended signal."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite values.")

    nperseg = min(
        int(round(segment_seconds * fs)),
        len(values),
    )

    if nperseg < 16:
        raise ValueError("Analysis block is too short.")

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


def solve_wavenumber(
    frequency_hz,
    water_depth_m,
    gravity=9.81,
    tolerance=1e-12,
    maximum_iterations=100,
):
    """
    Solve omega^2 = g k tanh(kh) by Newton iteration.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency

    shallow_guess = omega / np.sqrt(gravity * water_depth_m)
    deep_guess = omega**2 / gravity
    k = np.maximum(shallow_guess, deep_guess)
    k[omega == 0] = 0.0

    for _ in range(maximum_iterations):
        kh = k * water_depth_m
        tanh_kh = np.tanh(kh)

        function = gravity * k * tanh_kh - omega**2
        # sech²(kh) = 1 - tanh²(kh), evaluated without cosh overflow.
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
        update[valid] = function[valid] / derivative[valid]

        new_k = np.maximum(k - update, 0.0)

        if np.nanmax(np.abs(new_k - k)) < tolerance:
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
            "Estimated water depth must exceed pressure-sensor height."
        )

    frequency = np.asarray(frequency_hz, dtype=np.float64)

    k = solve_wavenumber(
        frequency_hz=frequency,
        water_depth_m=water_depth_m,
        gravity=gravity,
    )

    kz = k * sensor_height_above_bed_m
    kh = k * water_depth_m
    
    # Stable equivalent of cosh(kz) / cosh(kh):
    #
    # cosh(a) / cosh(b)
    # = exp(a-b) * [1 + exp(-2a)] / [1 + exp(-2b)]
    #
    # Here kh >= kz >= 0, so no positive exponential becomes large.
    kp = (
        np.exp(kz - kh)
        * (1.0 + np.exp(-2.0 * kz))
        / (1.0 + np.exp(-2.0 * kh))
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

    Unreliable bins are returned as NaN.
    """
    kp, wavenumber = pressure_response_factor(
        frequency_hz=frequency,
        water_depth_m=water_depth_m,
        sensor_height_above_bed_m=sensor_height_above_bed_m,
        gravity=gravity,
    )

    amplitude_gain = np.full_like(
    kp,
    np.inf,
    dtype=np.float64,
    )

    # A bin is only numerically usable if Kp is finite and large enough
    # that its reciprocal does not overflow.
    minimum_kp = (
        1.0 / maximum_amplitude_gain
    )
    
    valid_kp = (
        np.isfinite(kp)
        & (kp >= minimum_kp)
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
        & (amplitude_gain <= maximum_amplitude_gain)
    )

    eta_psd = np.full_like(pressure_head_psd, np.nan)
    eta_psd[reliable] = (
        pressure_head_psd[reliable]
        / kp[reliable] ** 2
    )

    return eta_psd, kp, amplitude_gain, reliable, wavenumber


def contiguous_reliable_upper_frequency(
    frequency,
    reliable,
    lower_hz,
    requested_upper_hz,
):
    """
    Return the upper edge of the contiguous reliable interval beginning
    at lower_hz.
    """
    frequency = np.asarray(frequency, dtype=float)
    reliable = np.asarray(reliable, dtype=bool)

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

    df = np.nanmedian(np.diff(frequency))

    return min(
        requested_upper_hz,
        frequency[accepted_indices[-1]] + df,
    )


def integrate_spectral_band(
    frequency,
    psd,
    lower_hz,
    upper_hz,
):
    """Integrate PSD over [lower_hz, upper_hz)."""
    frequency = np.asarray(frequency, dtype=float)
    psd = np.asarray(psd, dtype=float)

    selected = (
        np.isfinite(frequency)
        & np.isfinite(psd)
        & (frequency >= lower_hz)
        & (frequency < upper_hz)
    )

    if selected.sum() < 2:
        return np.nan

    selected_frequency = frequency[selected]
    selected_psd = psd[selected]

    df = np.nanmedian(np.diff(frequency))

    # Avoid integrating across an internal unreliable frequency gap.
    if np.any(np.diff(selected_frequency) > 1.5 * df):
        return np.nan

    return np.trapezoid(
        selected_psd,
        selected_frequency,
    )


# ============================================================
# READ PRESSURE PARQUET
# ============================================================

print("Reading:")
print(input_file)

df = pd.read_parquet(input_file)

required_columns = {"time", pressure_column}
missing = required_columns - set(df.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

df = df[["time", pressure_column]].copy()

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
    df.dropna(subset=["time"])
    .sort_values("time")
    .drop_duplicates(subset=["time"], keep="first")
    .reset_index(drop=True)
)

print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows: ", len(df))
print("\nMissing pressure values:")
print(df[pressure_column].isna().sum())


# ============================================================
# DETECT NATURAL BURSTS AND NOMINAL LENGTH
# ============================================================

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

print("\nContinuous segments:", len(segment_summary))
print(
    "Detected nominal burst:",
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


# ============================================================
# CREATE ANALYSIS BLOCKS
# ============================================================

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

print("Analysis blocks produced:", len(analysis_blocks))


# ============================================================
# COMPUTE SPECTRA AND WAVE HEIGHTS
# ============================================================

statistics_records = []
spectral_records = []

for analysis_block_number, block in enumerate(analysis_blocks):
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

    valid_fraction = (
        block[pressure_column]
        .notna()
        .mean()
    )

    rejection_reason = ""

    if sample_count < minimum_block_samples:
        rejection_reason = "analysis block too short"
    elif valid_fraction < minimum_valid_fraction:
        rejection_reason = "too many missing pressure samples"

    base_record = {
        "analysis_block_number": analysis_block_number,
        "source_segment_id": source_segment_id,
        "subblock_number": subblock_number,
        "start_time": start_time,
        "end_time": end_time,
        "sample_count": sample_count,
        "duration_minutes": duration_minutes,
        "valid_fraction": valid_fraction,
    }

    if rejection_reason:
        statistics_records.append(
            {
                **base_record,
                "accepted": False,
                "rejection_reason": rejection_reason,
                "mean_pressure_pa": np.nan,
                "mean_pressure_head_m": np.nan,
                "mean_water_depth_m": np.nan,
                "ig_variance_m2": np.nan,
                "ss_variance_m2": np.nan,
                "ss_common_fc_variance_m2": np.nan,
                "hm0_ig_m": np.nan,
                "hm0_ss_m": np.nan,
                "hm0_ss_common_fc_m": np.nan,
                "common_ss_high_hz": common_ss_high_hz,
                "common_fc_reliable": False,
                "ig_to_ss_variance_ratio": np.nan,
                "ig_effective_high_hz": np.nan,
                "ss_effective_high_hz": np.nan,
                "max_reliable_frequency_hz": np.nan,
            }
        )
        continue

    pressure_pa = (
        block[pressure_column]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
        .to_numpy(dtype=np.float64)
    )

    mean_pressure_pa = float(np.mean(pressure_pa))

    # P_APC is pressure due to seawater, in Pa.
    pressure_head_m = (
        pressure_pa
        / (rho_water_kg_m3 * gravity_m_s2)
    )

    mean_pressure_head_m = float(
        np.mean(pressure_head_m)
    )

    # Hydrostatic relation:
    # P/(rho*g) = h - z_sensor
    mean_water_depth_m = (
        mean_pressure_head_m
        + z_pressure_m
    )

    if (
        not np.isfinite(mean_water_depth_m)
        or mean_water_depth_m < minimum_water_depth_m
        or mean_water_depth_m <= z_pressure_m
    ):
        statistics_records.append(
            {
                **base_record,
                "accepted": False,
                "rejection_reason":
                    "invalid estimated mean water depth",
                "mean_pressure_pa": mean_pressure_pa,
                "mean_pressure_head_m": mean_pressure_head_m,
                "mean_water_depth_m": mean_water_depth_m,
                "ig_variance_m2": np.nan,
                "ss_variance_m2": np.nan,
                "ss_common_fc_variance_m2": np.nan,
                "hm0_ig_m": np.nan,
                "hm0_ss_m": np.nan,
                "hm0_ss_common_fc_m": np.nan,
                "common_ss_high_hz": common_ss_high_hz,
                "common_fc_reliable": False,
                "ig_to_ss_variance_ratio": np.nan,
                "ig_effective_high_hz": np.nan,
                "ss_effective_high_hz": np.nan,
                "max_reliable_frequency_hz": np.nan,
            }
        )
        continue
    
    # Deployment-specific QC:
    # reject pressure offsets that imply an implausible depth.
    if (
    abs(mean_water_depth_m - expected_water_depth_m) > maximum_depth_deviation_m
    ):
        statistics_records.append(
            {
                **base_record,
                "accepted": False,
                "rejection_reason":
                    "implausible mean water depth",
                "mean_pressure_pa": mean_pressure_pa,
                "mean_pressure_head_m": mean_pressure_head_m,
                "mean_water_depth_m": mean_water_depth_m,
                "ig_variance_m2": np.nan,
                "ss_variance_m2": np.nan,
                "ss_common_fc_variance_m2": np.nan,
                "hm0_ig_m": np.nan,
                "hm0_ss_m": np.nan,
                "hm0_ss_common_fc_m": np.nan,
                "common_ss_high_hz": common_ss_high_hz,
                "common_fc_reliable": False,
                "ig_to_ss_variance_ratio": np.nan,
                "ig_effective_high_hz": np.nan,
                "ss_effective_high_hz": np.nan,
                "max_reliable_frequency_hz": np.nan,
            }
        )
        continue

    # Remove tide, surge, and slow water-level variation.
    pressure_head_anomaly, pressure_head_background = (
        polynomial_detrend(
            pressure_head_m,
            order=detrend_order,
        )
    )

    frequency, pressure_head_psd = calculate_autospectrum(
        values=pressure_head_anomaly,
        fs=fs,
        segment_seconds=welch_segment_seconds,
        overlap_fraction=overlap_fraction,
    )

    (
        eta_psd,
        kp,
        amplitude_gain,
        reliable,
        wavenumber,
    ) = pressure_to_surface_spectrum(
        frequency=frequency,
        pressure_head_psd=pressure_head_psd,
        water_depth_m=mean_water_depth_m,
        sensor_height_above_bed_m=z_pressure_m,
        gravity=gravity_m_s2,
        maximum_amplitude_gain=max_pressure_amplitude_gain,
    )

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
        reliable & (frequency > 0)
    ]

    max_reliable_frequency_hz = (
        float(reliable_positive.max())
        if len(reliable_positive)
        else np.nan
    )

    ig_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ig_low_hz,
            min(ig_high_hz, ig_effective_high_hz),
        )
        if np.isfinite(ig_effective_high_hz)
        else np.nan
    )

    ss_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ss_low_hz,
            min(ss_high_hz, ss_effective_high_hz),
        )
        if np.isfinite(ss_effective_high_hz)
        else np.nan
    )

    # Sea-swell variance over the fixed common band used to compare frames.
    # It is accepted only when the gain-limited reliable interval reaches the
    # complete common band.
    common_fc_reliable = (
        np.isfinite(ss_effective_high_hz)
        and ss_effective_high_hz >= common_ss_high_hz
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

    hm0_ig_m = (
        4.0 * np.sqrt(ig_variance_m2)
        if np.isfinite(ig_variance_m2)
        else np.nan
    )

    hm0_ss_m = (
        4.0 * np.sqrt(ss_variance_m2)
        if np.isfinite(ss_variance_m2)
        else np.nan
    )

    hm0_ss_common_fc_m = (
        4.0 * np.sqrt(ss_common_fc_variance_m2)
        if np.isfinite(ss_common_fc_variance_m2)
        else np.nan
    )

    ig_to_ss_variance_ratio = (
        ig_variance_m2 / ss_variance_m2
        if (
            np.isfinite(ss_variance_m2)
            and ss_variance_m2 > 0
        )
        else np.nan
    )

    ig_to_ss_common_fc_variance_ratio = (
        ig_variance_m2 / ss_common_fc_variance_m2
        if (
            np.isfinite(ss_common_fc_variance_m2)
            and ss_common_fc_variance_m2 > 0
        )
        else np.nan
    )

    statistics_records.append(
        {
            **base_record,
            "accepted": True,
            "rejection_reason": "",
            "mean_pressure_pa": mean_pressure_pa,
            "mean_pressure_head_m": mean_pressure_head_m,
            "mean_water_depth_m": mean_water_depth_m,
            "ig_variance_m2": ig_variance_m2,
            "ss_variance_m2": ss_variance_m2,
            "ss_common_fc_variance_m2":
                ss_common_fc_variance_m2,
            "hm0_ig_m": hm0_ig_m,
            "hm0_ss_m": hm0_ss_m,
            "hm0_ss_common_fc_m": hm0_ss_common_fc_m,
            "common_ss_high_hz": common_ss_high_hz,
            "common_fc_reliable": common_fc_reliable,
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
                "start_time":
                    start_time,
                "end_time":
                    end_time,
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
            }
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

statistics = pd.DataFrame(statistics_records)

spectra = (
    pd.concat(spectral_records, ignore_index=True)
    if spectral_records
    else pd.DataFrame()
)

statistics_file = (
    output_folder
    / f"{case_id}_pressure_burst_statistics.csv"
)

spectra_file = (
    output_folder
    / f"{case_id}_surface_elevation_spectra.pkl"
)

segment_file = (
    output_folder
    / f"{case_id}_pressure_segment_summary.csv"
)

# One shared, long-format table for comparison across all cases.
# Re-running a case replaces that case's existing rows rather than duplicating
# them.
common_wave_heights_file = (
    output_folder
    / "all_cases_wave_heights_common_fc.csv"
)

statistics.to_csv(statistics_file, index=False)
spectra.to_pickle(spectra_file)
segment_summary.to_csv(segment_file, index=False)

comparison_columns = [
    "case_id",
    "case_label",
    "analysis_block_number",
    "start_time",
    "end_time",
    "mid_time",
    "mean_water_depth_m",
    "ig_low_hz",
    "ig_high_hz",
    "ss_low_hz",
    "common_ss_high_hz",
    "ss_effective_high_hz",
    "common_fc_reliable",
    "hm0_ig_m",
    "hm0_ss_common_fc_m",
    "hm0_ss_m",
]

comparison = statistics.loc[statistics["accepted"]].copy()
comparison["mid_time"] = (
    comparison["start_time"]
    + (comparison["end_time"] - comparison["start_time"]) / 2
)
comparison["case_id"] = case_id
comparison["case_label"] = case_label
comparison["ig_low_hz"] = ig_low_hz
comparison["ig_high_hz"] = ig_high_hz
comparison["ss_low_hz"] = ss_low_hz
comparison = comparison[comparison_columns]

if common_wave_heights_file.exists():
    existing_comparison = pd.read_csv(
        common_wave_heights_file,
        parse_dates=["start_time", "end_time", "mid_time"],
    )
    if "case_id" not in existing_comparison.columns:
        raise KeyError(
            f"Existing shared file lacks case_id: {common_wave_heights_file}"
        )
    existing_comparison = existing_comparison.loc[
        existing_comparison["case_id"] != case_id
    ]
    comparison = pd.concat(
        [existing_comparison, comparison],
        ignore_index=True,
    )

comparison = comparison.sort_values(
    ["case_id", "start_time", "analysis_block_number"]
)
comparison.to_csv(common_wave_heights_file, index=False)

print("\nSaved segment summary:")
print(segment_file)

print("\nSaved spectral statistics:")
print(statistics_file)

print("\nSaved spectra:")
print(spectra_file)

print("\nUpdated shared cross-case wave-height file:")
print(common_wave_heights_file)

print("\nAccepted blocks:")
print(statistics["accepted"].sum())

print("Rejected blocks:")
print((~statistics["accepted"]).sum())


# ============================================================
# SUMMARY AND TIME SERIES
# ============================================================

accepted = statistics.loc[
    statistics["accepted"]
].copy()

if accepted.empty:
    raise RuntimeError(
        "No pressure blocks passed the quality checks."
    )

accepted["mid_time"] = (
    accepted["start_time"]
    + (
        accepted["end_time"]
        - accepted["start_time"]
    ) / 2
)

accepted = accepted.sort_values("mid_time")

print("\nAccepted-burst statistics:")
print(
    accepted[
        [
            "mean_water_depth_m",
            "hm0_ig_m",
            "hm0_ss_m",
            "hm0_ss_common_fc_m",
            "ig_to_ss_variance_ratio",
            "ig_to_ss_common_fc_variance_ratio",
            "ig_effective_high_hz",
            "ss_effective_high_hz",
        ]
    ].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
)

print(
    f"\nHm0_SS_common_fc uses the fixed comparison band "
    f"{ss_low_hz:.3f}-{common_ss_high_hz:.3f} Hz. Values are NaN "
    "when the automatically detected gain-limited cutoff does not reach "
    "the full common band."
)

print(
    "\nHm0_SS must be interpreted together with "
    "ss_effective_high_hz. When this cutoff is below 1 Hz, "
    "Hm0_SS is the pressure-recoverable part of the requested "
    "0.05-1.00 Hz band, not the complete sea-swell wave height."
)

dt = accepted["mid_time"].diff().dt.total_seconds()

accepted["hm0_ig_plot"] = accepted["hm0_ig_m"]
accepted["hm0_ss_plot"] = accepted["hm0_ss_m"]
accepted["hm0_ss_common_fc_plot"] = accepted["hm0_ss_common_fc_m"]

large_gap = dt > 3600

accepted.loc[
    large_gap,
    ["hm0_ig_plot", "hm0_ss_plot", "hm0_ss_common_fc_plot"],
] = np.nan

plt.figure(figsize=(12, 5))

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ig_plot"],
    label=r"$H_{m0,IG}$",
)

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ss_plot"],
    label=r"$H_{m0,SS}$ (automatic cutoff)",
)

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ss_common_fc_plot"],
    label=rf"$H_{{m0,SS}}$ (common $f_c$={common_ss_high_hz:.3f} Hz)",
)

plt.xlabel("Time")
plt.ylabel("Spectral wave height (m)")
plt.title(case_label)
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================
# MOST ENERGETIC IG BURST
# ============================================================

energetic_candidates = accepted.dropna(
    subset=["ig_variance_m2"]
)

if not energetic_candidates.empty and not spectra.empty:
    most_energetic_row = energetic_candidates.loc[
        energetic_candidates["ig_variance_m2"].idxmax()
    ]

    block_number = int(
        most_energetic_row["analysis_block_number"]
    )

    selected = spectra.loc[
        spectra["analysis_block_number"] == block_number
    ].copy()

    plot_data = selected.loc[
        (selected["frequency_hz"] > 0)
        & selected["eta_psd_m2_hz"].notna()
    ]

    plt.figure(figsize=(9, 6))

    plt.loglog(
        plot_data["frequency_hz"],
        plot_data["eta_psd_m2_hz"],
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
        label="Requested sea-swell band",
    )

    cutoff = most_energetic_row[
        "ss_effective_high_hz"
    ]

    plt.axvline(
        common_ss_high_hz,
        linestyle=":",
        label=f"Common cutoff ({common_ss_high_hz:.3f} Hz)",
    )

    if np.isfinite(cutoff):
        plt.axvline(
            cutoff,
            linestyle="--",
            label=f"Reliable cutoff ({cutoff:.3f} Hz)",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Surface-elevation PSD (m$^2$/Hz)")

    plt.title(
        f"Most energetic IG burst ({case_label})\n"
        f"{most_energetic_row['start_time']} to "
        f"{most_energetic_row['end_time']}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MEDIAN SPECTRUM
# ============================================================

if not spectra.empty:
    median_spectrum = (
        spectra.groupby(
            "frequency_hz",
            as_index=False,
        )
        .agg(
            eta_psd_median=(
                "eta_psd_m2_hz",
                "median",
            ),
            eta_psd_q25=(
                "eta_psd_m2_hz",
                lambda x: x.quantile(0.25),
            ),
            eta_psd_q75=(
                "eta_psd_m2_hz",
                lambda x: x.quantile(0.75),
            ),
        )
    )

    plot_data = median_spectrum.loc[
        (median_spectrum["frequency_hz"] > 0)
        & median_spectrum["eta_psd_median"].notna()
    ]

    plt.figure(figsize=(9, 6))

    plt.loglog(
        plot_data["frequency_hz"],
        plot_data["eta_psd_median"],
        label="Median",
    )

    plt.fill_between(
        plot_data["frequency_hz"],
        plot_data["eta_psd_q25"],
        plot_data["eta_psd_q75"],
        alpha=0.2,
        label="Interquartile range",
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
        label="Requested sea-swell band",
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Surface-elevation PSD (m$^2$/Hz)")
    plt.title(
        f"Median surface-elevation spectrum ({case_label})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MEAN SPECTRUM
# ============================================================

if not spectra.empty:
    mean_spectrum = (
        spectra.groupby(
            "frequency_hz",
            as_index=False,
        )
        .agg(
            eta_psd_mean=(
                "eta_psd_m2_hz",
                "mean",
            ),
            eta_psd_std=(
                "eta_psd_m2_hz",
                "std",
            ),
        )
    )

    plot_data = mean_spectrum.loc[
        (mean_spectrum["frequency_hz"] > 0)
        & mean_spectrum["eta_psd_mean"].notna()
    ]

    plt.figure(figsize=(9, 6))

    plt.loglog(
        plot_data["frequency_hz"],
        plot_data["eta_psd_mean"],
        label="Mean",
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
        label="Requested sea-swell band",
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Surface-elevation PSD (m$^2$/Hz)")
    plt.title(
        f"Mean surface-elevation spectrum ({case_label})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
