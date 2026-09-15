# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:08:28 2026

@author: WangX3

Compute pressure-derived surface-elevation autospectra and band-limited
spectral wave heights from a processed upward-looking ADCP pressure Parquet file.

The workflow mirrors the velocity script:
1. detect natural continuous bursts from timestamp gaps;
2. detect the nominal burst length;
3. split long segments into nominal analysis blocks;
4. convert atmospheric-pressure-corrected pressure to pressure head;
5. estimate mean water depth;
6. detrend each block;
7. calculate a Welch autospectrum;
8. correct for depth attenuation using linear wave theory up to one
   fixed candidate cutoff frequency;
9. check Kp and 1/Kp at that candidate cutoff for every burst;
10. append an equilibrium f^-4 tail above the cutoff;
11. calculate Hm0,IG and direct/tail-extended Hm0,SS.

Important: the fixed cutoff is used only for bursts where the amplitude
gain 1/Kp at candidate_fc_hz does not exceed max_pressure_amplitude_gain.
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
    r"\Pressure_DVN_F1_ADCP_UP.parquet"
)

case_id = "DVN_F1_ADCP_UP"
case_label = "DVN F1 ADCP UP"

pressure_column = "pressure_corrected"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)
output_folder.mkdir(parents=True, exist_ok=True)

# ADCP pressure is already stored at its native sampling frequency.
# The pressure-sensor height above bed is read from the Parquet file;
# do not set z_pressure_m manually.
pressure_units = "Pa"

rho_water_kg_m3 = 1025.0
gravity_m_s2 = 9.81

gap_threshold_seconds = 1.0

welch_segment_seconds = 512.0
overlap_fraction = 0.5

ig_low_hz = 0.005
ig_high_hz = 0.04

ss_low_hz = 0.04
ss_high_hz = 1.0

# Fixed candidate pressure-correction cutoff for all bursts.
# Below this frequency, surface elevation is reconstructed directly
# from pressure using the linear pressure-response factor Kp.
# Above this frequency, the unresolved sea-swell spectrum is extended
# with an equilibrium f^-4 tail.
candidate_fc_hz = 0.20

# Maximum acceptable free-surface amplitude gain at candidate_fc_hz:
#     gain = 1 / Kp
# A burst is considered safe for the fixed cutoff only when gain <= this.
max_pressure_amplitude_gain = 15.0

# High-frequency equilibrium-tail settings.
tail_exponent = 4.0
tail_anchor_bins = 5

minimum_valid_fraction = 0.98
minimum_block_fraction = 0.95
minimum_water_depth_m = 0.25
expected_water_depth_m = 20.0  # change only when processing another frame
maximum_depth_deviation_m = 3.0

detrend_order = 2

if not (ss_low_hz < candidate_fc_hz < ss_high_hz):
    raise ValueError(
        "candidate_fc_hz must lie strictly inside the requested "
        "sea-swell band."
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


def build_fixed_fc_surface_spectrum(
    frequency,
    pressure_head_psd,
    water_depth_m,
    sensor_height_above_bed_m,
    candidate_fc_hz,
    ss_high_hz,
    maximum_amplitude_gain=10.0,
    tail_exponent=4.0,
    tail_anchor_bins=5,
    gravity=9.81,
):
    """
    Reconstruct surface elevation with one fixed cutoff for all bursts.

    1. Compute Kp and gain = 1/Kp over the full frequency vector.
    2. Evaluate Kp and gain at candidate_fc_hz.
    3. If gain(candidate_fc_hz) <= maximum_amplitude_gain, reconstruct
       S_eta = S_pressure_head / Kp^2 for f < candidate_fc_hz.
    4. For candidate_fc_hz <= f < ss_high_hz, append an equilibrium
       f^(-tail_exponent) tail.

    The tail level is estimated robustly from the last several directly
    reconstructed bins below candidate_fc_hz. The equivalent spectrum
    at the cutoff is

        S_fc = median[S_eta(f_i) * (f_i / fc)^tail_exponent]

    so that

        S_tail(f) = S_fc * (f / fc)^(-tail_exponent).

    Returns
    -------
    eta_psd_direct
        Direct pressure-derived surface-elevation spectrum below fc.
    eta_psd_extended
        Direct spectrum below fc + equilibrium tail above fc.
    eta_psd_tail
        Tail only; NaN outside the tail interval.
    kp
        Pressure response factor at every frequency.
    amplitude_gain
        1/Kp at every frequency.
    candidate_kp
        Kp evaluated at candidate_fc_hz.
    candidate_gain
        1/Kp evaluated at candidate_fc_hz.
    candidate_fc_safe
        True when candidate_gain <= maximum_amplitude_gain.
    tail_level_at_fc
        Estimated S_eta(fc) used to anchor the f^-4 tail.
    wavenumber
        Wavenumber at every frequency.
    """
    frequency = np.asarray(frequency, dtype=float)
    pressure_head_psd = np.asarray(pressure_head_psd, dtype=float)

    kp, wavenumber = pressure_response_factor(
        frequency_hz=frequency,
        water_depth_m=water_depth_m,
        sensor_height_above_bed_m=sensor_height_above_bed_m,
        gravity=gravity,
    )

    amplitude_gain = np.full_like(kp, np.inf, dtype=float)
    valid_kp = np.isfinite(kp) & (kp > 0)

    np.divide(
        1.0,
        kp,
        out=amplitude_gain,
        where=valid_kp,
    )

    # Evaluate the response factor exactly at the fixed candidate fc
    candidate_kp_array, _ = pressure_response_factor(
        frequency_hz=np.array([candidate_fc_hz], dtype=float),
        water_depth_m=water_depth_m,
        sensor_height_above_bed_m=sensor_height_above_bed_m,
        gravity=gravity,
    )

    candidate_kp = float(candidate_kp_array[0])
    candidate_gain = (
        1.0 / candidate_kp
        if np.isfinite(candidate_kp) and candidate_kp > 0
        else np.inf
    )

    candidate_fc_safe = bool(
        np.isfinite(candidate_gain)
        and candidate_gain <= maximum_amplitude_gain
    )

    eta_psd_direct = np.full_like(
        pressure_head_psd,
        np.nan,
        dtype=float,
    )
    eta_psd_extended = np.full_like(
        pressure_head_psd,
        np.nan,
        dtype=float,
    )
    eta_psd_tail = np.full_like(
        pressure_head_psd,
        np.nan,
        dtype=float,
    )

    # Do not use the fixed cutoff if the pressure amplification at fc
    # fails the selected safety criterion.
    if not candidate_fc_safe:
        return (
            eta_psd_direct,
            eta_psd_extended,
            eta_psd_tail,
            kp,
            amplitude_gain,
            candidate_kp,
            candidate_gain,
            candidate_fc_safe,
            np.nan,
            wavenumber,
        )

    direct_mask = (
        np.isfinite(frequency)
        & np.isfinite(pressure_head_psd)
        & np.isfinite(kp)
        & (kp > 0)
        & (frequency >= 0)
        & (frequency < candidate_fc_hz)
    )

    eta_psd_direct[direct_mask] = (
        pressure_head_psd[direct_mask]
        / kp[direct_mask] ** 2
    )
    eta_psd_extended[direct_mask] = eta_psd_direct[direct_mask]

    # Anchor the equilibrium tail using the last several finite bins
    # immediately below fc. This is less sensitive to one noisy bin.
    anchor_idx = np.flatnonzero(
        direct_mask
        & np.isfinite(eta_psd_direct)
        & (eta_psd_direct > 0)
        & (frequency >= ss_low_hz)
    )

    if anchor_idx.size < 2:
        return (
            eta_psd_direct,
            eta_psd_extended,
            eta_psd_tail,
            kp,
            amplitude_gain,
            candidate_kp,
            candidate_gain,
            candidate_fc_safe,
            np.nan,
            wavenumber,
        )

    anchor_idx = anchor_idx[-max(2, int(tail_anchor_bins)):]

    anchor_f = frequency[anchor_idx]
    anchor_S = eta_psd_direct[anchor_idx]

    # Estimate the equivalent spectral density at exactly fc.
    tail_level_at_fc = float(
        np.nanmedian(
            anchor_S
            * (anchor_f / candidate_fc_hz) ** tail_exponent
        )
    )

    if not np.isfinite(tail_level_at_fc) or tail_level_at_fc <= 0:
        return (
            eta_psd_direct,
            eta_psd_extended,
            eta_psd_tail,
            kp,
            amplitude_gain,
            candidate_kp,
            candidate_gain,
            candidate_fc_safe,
            np.nan,
            wavenumber,
        )

    tail_mask = (
        np.isfinite(frequency)
        & (frequency >= candidate_fc_hz)
        & (frequency < ss_high_hz)
        & (frequency > 0)
    )

    eta_psd_tail[tail_mask] = (
        tail_level_at_fc
        * (frequency[tail_mask] / candidate_fc_hz) ** (-tail_exponent)
    )

    eta_psd_extended[tail_mask] = eta_psd_tail[tail_mask]

    return (
        eta_psd_direct,
        eta_psd_extended,
        eta_psd_tail,
        kp,
        amplitude_gain,
        candidate_kp,
        candidate_gain,
        candidate_fc_safe,
        tail_level_at_fc,
        wavenumber,
    )


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

def extract_sensor_depth_m(df, pressure_column):
    """Read pressure-sensor height above bed from the Parquet content/metadata.

    Supported layouts, in order:
    1. a scalar/constant ``sensor_depth`` column;
    2. ``<pressure_column>_sensor_depth`` column;
    3. pandas attrs: df.attrs[pressure_column]["sensor_depth"];
    4. pandas attrs: df.attrs["sensor_depth"].

    The function deliberately fails rather than silently using a manual value.
    """
    candidates = ["sensor_depth", f"{pressure_column}_sensor_depth"]
    for col in candidates:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if not x.empty:
                z = float(x.median())
                if z > 0:
                    return z, f"column {col!r}"

    meta = df.attrs.get(pressure_column, None)
    if isinstance(meta, dict) and "sensor_depth" in meta:
        z = float(meta["sensor_depth"])
        if np.isfinite(z) and z > 0:
            return z, f"df.attrs[{pressure_column!r}]['sensor_depth']"

    if "sensor_depth" in df.attrs:
        z = float(df.attrs["sensor_depth"])
        if np.isfinite(z) and z > 0:
            return z, "df.attrs['sensor_depth']"

    raise KeyError(
        "Could not read sensor_depth from the pressure Parquet file. "
        "Expected a 'sensor_depth' column, a "
        f"'{pressure_column}_sensor_depth' column, or corresponding pandas attrs."
    )


def pressure_to_pa(values, units):
    """Convert corrected pressure to Pa."""
    units = str(units).strip().lower()
    values = np.asarray(values, dtype=np.float64)
    factors = {
        "pa": 1.0, "pascal": 1.0, "pascals": 1.0,
        "kpa": 1e3, "kilopascal": 1e3, "kilopascals": 1e3,
        "hpa": 1e2, "mbar": 1e2, "millibar": 1e2,
        "bar": 1e5, "bars": 1e5,
        "dbar": 1e4, "decibar": 1e4, "decibars": 1e4,
    }
    if units not in factors:
        raise ValueError(f"Unrecognized pressure units {units!r}.")
    return values * factors[units]


# ============================================================
# READ ADCP PRESSURE PARQUET
# ============================================================

print("Reading:")
print(input_file)

df = pd.read_parquet(input_file)

required_columns = {"time", pressure_column}
missing = required_columns - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

# Read sensor depth BEFORE subsetting columns, because it may be stored
# as a dedicated Parquet column. pandas attrs are also checked.
z_pressure_m, sensor_depth_source = extract_sensor_depth_m(df, pressure_column)
print(f"Pressure sensor height above bed = {z_pressure_m:.3f} m")
print(f"Sensor-depth source: {sensor_depth_source}")

keep_columns = ["time", pressure_column]
for optional_col in ["sensor_depth", f"{pressure_column}_sensor_depth"]:
    if optional_col in df.columns:
        keep_columns.append(optional_col)
df = df[keep_columns].copy()

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df[pressure_column] = pd.to_numeric(df[pressure_column], errors="coerce")
df = (
    df.dropna(subset=["time"])
      .sort_values("time")
      .drop_duplicates(subset=["time"], keep="first")
      .reset_index(drop=True)
)

print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows: ", len(df))
print("Missing pressure values:", df[pressure_column].isna().sum())

# ADCP pressure is already at native sampling: detect fs directly.
fs, regular_dt_seconds = detect_sampling_frequency(df["time"])
gap_threshold_seconds = max(3.0 * regular_dt_seconds, 2.0)

print("\nDetected ADCP pressure sampling:")
print("dt =", regular_dt_seconds, "s")
print("fs =", fs, "Hz")
print("Gap threshold =", gap_threshold_seconds, "s")

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
directional_pressure_records = []

for analysis_block_number, block in enumerate(analysis_blocks):

    start_time = block["time"].iloc[0]
    end_time = block["time"].iloc[-1]
    mid_time = start_time + (end_time - start_time) / 2

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

    # --------------------------------------------------------
    # Helper for recording rejected blocks
    # --------------------------------------------------------

    def rejected_record(
        reason,
        mean_pressure_pa=np.nan,
        mean_pressure_head_m=np.nan,
        mean_water_depth_m=np.nan,
        pressure_std_pa=np.nan,
        pressure_frozen_fraction=np.nan,
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
            "pressure_std_pa":
                pressure_std_pa,
            "pressure_frozen_fraction":
                pressure_frozen_fraction,
            "ig_variance_m2":
                np.nan,
            "ss_direct_variance_m2":
                np.nan,
            "ss_tail_variance_m2":
                np.nan,
            "ss_total_variance_m2":
                np.nan,
            "hm0_ig_m":
                np.nan,
            "hm0_ss_direct_m":
                np.nan,
            "hm0_ss_total_m":
                np.nan,
            "candidate_fc_hz":
                candidate_fc_hz,
            "kp_at_candidate_fc":
                np.nan,
            "gain_at_candidate_fc":
                np.nan,
            "candidate_fc_safe":
                False,
            "tail_level_at_fc_m2_hz":
                np.nan,
            "ig_to_ss_total_variance_ratio":
                np.nan,
        }

    # --------------------------------------------------------
    # Basic block QC
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
    # Fill isolated missing pressure values
    # --------------------------------------------------------

    pressure_native = (
        block[pressure_column]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(pressure_native).all():
        statistics_records.append(
            rejected_record(
                "non-finite pressure after interpolation"
            )
        )
        continue

    pressure_pa = pressure_to_pa(pressure_native, pressure_units)

    mean_pressure_pa = float(
        np.mean(pressure_pa)
    )

    # --------------------------------------------------------
    # Check for frozen / near-constant pressure
    # --------------------------------------------------------

    pressure_std_pa = float(
        np.std(pressure_pa)
    )

    if len(pressure_pa) > 1:
        pressure_frozen_fraction = float(
            np.mean(
                np.diff(pressure_pa) == 0
            )
        )
    else:
        pressure_frozen_fraction = np.nan

    # Reject only essentially completely frozen records.
    # Do not use a large arbitrary std threshold because repeated
    # pressure values can legitimately occur due to quantization.
    if (
        not np.isfinite(pressure_std_pa)
        or pressure_std_pa <= 0
        or (
            np.isfinite(pressure_frozen_fraction)
            and pressure_frozen_fraction > 0.99
        )
    ):
        statistics_records.append(
            rejected_record(
                "frozen or near-constant pressure signal",
                mean_pressure_pa=mean_pressure_pa,
                pressure_std_pa=pressure_std_pa,
                pressure_frozen_fraction=
                    pressure_frozen_fraction,
            )
        )
        continue

    # --------------------------------------------------------
    # Convert pressure to pressure head and water depth
    # --------------------------------------------------------

    pressure_head_m = (
        pressure_pa
        / (rho_water_kg_m3 * gravity_m_s2)
    )

    mean_pressure_head_m = float(
        np.mean(pressure_head_m)
    )

    # Hydrostatic relation:
    #
    # P/(rho*g) = h - z_sensor
    #
    # therefore:
    #
    # h = P/(rho*g) + z_sensor

    mean_water_depth_m = (
        mean_pressure_head_m
        + z_pressure_m
    )

    # --------------------------------------------------------
    # Water-depth QC
    # --------------------------------------------------------

    if (
        not np.isfinite(mean_water_depth_m)
        or mean_water_depth_m < minimum_water_depth_m
        or mean_water_depth_m <= z_pressure_m
    ):
        statistics_records.append(
            rejected_record(
                "invalid estimated mean water depth",
                mean_pressure_pa=mean_pressure_pa,
                mean_pressure_head_m=
                    mean_pressure_head_m,
                mean_water_depth_m=
                    mean_water_depth_m,
                pressure_std_pa=
                    pressure_std_pa,
                pressure_frozen_fraction=
                    pressure_frozen_fraction,
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
                mean_pressure_pa=mean_pressure_pa,
                mean_pressure_head_m=
                    mean_pressure_head_m,
                mean_water_depth_m=
                    mean_water_depth_m,
                pressure_std_pa=
                    pressure_std_pa,
                pressure_frozen_fraction=
                    pressure_frozen_fraction,
            )
        )
        continue

    # --------------------------------------------------------
    # Detrend pressure head
    # --------------------------------------------------------

    pressure_head_anomaly, pressure_head_background = (
        polynomial_detrend(
            pressure_head_m,
            order=detrend_order,
        )
    )

    # --------------------------------------------------------
    # Calculate pressure-head PSD
    # --------------------------------------------------------

    frequency, pressure_head_psd = (
        calculate_autospectrum(
            values=pressure_head_anomaly,
            fs=fs,
            segment_seconds=
                welch_segment_seconds,
            overlap_fraction=
                overlap_fraction,
        )
    )

    # --------------------------------------------------------
    # Fixed-cutoff pressure correction + equilibrium f^-4 tail
    # --------------------------------------------------------

    (
        eta_psd,
        eta_psd_extended,
        eta_psd_tail,
        kp,
        amplitude_gain,
        kp_at_candidate_fc,
        gain_at_candidate_fc,
        candidate_fc_safe,
        tail_level_at_fc_m2_hz,
        wavenumber,
    ) = build_fixed_fc_surface_spectrum(
        frequency=frequency,
        pressure_head_psd=pressure_head_psd,
        water_depth_m=mean_water_depth_m,
        sensor_height_above_bed_m=z_pressure_m,
        candidate_fc_hz=candidate_fc_hz,
        ss_high_hz=ss_high_hz,
        maximum_amplitude_gain=max_pressure_amplitude_gain,
        tail_exponent=tail_exponent,
        tail_anchor_bins=tail_anchor_bins,
        gravity=gravity_m_s2,
    )

    # --------------------------------------------------------
    # Integrate IG variance
    # --------------------------------------------------------

    ig_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ig_low_hz,
            ig_high_hz,
        )
        if candidate_fc_safe
        else np.nan
    )

    # --------------------------------------------------------
    # Sea-swell variance below the fixed cutoff: directly pressure-derived
    # --------------------------------------------------------

    ss_direct_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd,
            ss_low_hz,
            candidate_fc_hz,
        )
        if candidate_fc_safe
        else np.nan
    )

    # --------------------------------------------------------
    # Sea-swell variance above the fixed cutoff: equilibrium f^-4 tail
    # --------------------------------------------------------

    ss_tail_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd_tail,
            candidate_fc_hz,
            ss_high_hz,
        )
        if (
            candidate_fc_safe
            and np.isfinite(tail_level_at_fc_m2_hz)
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Total SS variance over the requested 0.05-1 Hz band
    # --------------------------------------------------------

    ss_total_variance_m2 = (
        integrate_spectral_band(
            frequency,
            eta_psd_extended,
            ss_low_hz,
            ss_high_hz,
        )
        if (
            candidate_fc_safe
            and np.isfinite(tail_level_at_fc_m2_hz)
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Spectral QC
    #
    # A physically useful accepted block should not produce
    # zero/negative IG variance.
    # --------------------------------------------------------

    if (
        not np.isfinite(ig_variance_m2)
        or ig_variance_m2 <= 0
    ):
        statistics_records.append(
            rejected_record(
                "invalid or nonpositive IG spectral variance",
                mean_pressure_pa=mean_pressure_pa,
                mean_pressure_head_m=
                    mean_pressure_head_m,
                mean_water_depth_m=
                    mean_water_depth_m,
                pressure_std_pa=
                    pressure_std_pa,
                pressure_frozen_fraction=
                    pressure_frozen_fraction,
            )
        )
        continue

    # ============================================================
    # SAVE TIME-RESOLVED ADV PRESSURE FOR LATER
    # PRESSURE-VELOCITY CROSS-SPECTRAL ANALYSIS
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
    # Significant spectral wave heights
    # --------------------------------------------------------

    hm0_ig_m = (
        4.0 * np.sqrt(ig_variance_m2)
        if np.isfinite(ig_variance_m2) and ig_variance_m2 > 0
        else np.nan
    )

    hm0_ss_direct_m = (
        4.0 * np.sqrt(ss_direct_variance_m2)
        if (
            np.isfinite(ss_direct_variance_m2)
            and ss_direct_variance_m2 > 0
        )
        else np.nan
    )

    hm0_ss_total_m = (
        4.0 * np.sqrt(ss_total_variance_m2)
        if (
            np.isfinite(ss_total_variance_m2)
            and ss_total_variance_m2 > 0
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Variance ratio using the full SS estimate
    # --------------------------------------------------------

    ig_to_ss_total_variance_ratio = (
        ig_variance_m2 / ss_total_variance_m2
        if (
            np.isfinite(ss_total_variance_m2)
            and ss_total_variance_m2 > 0
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Save accepted block statistics
    # --------------------------------------------------------

    statistics_records.append(
        {
            **base_record,
            "accepted": True,
            "rejection_reason": "",
            "mean_pressure_pa":
                mean_pressure_pa,
            "mean_pressure_head_m":
                mean_pressure_head_m,
            "mean_water_depth_m":
                mean_water_depth_m,
            "pressure_std_pa":
                pressure_std_pa,
            "pressure_frozen_fraction":
                pressure_frozen_fraction,
            "ig_variance_m2":
                ig_variance_m2,
            "ss_direct_variance_m2":
                ss_direct_variance_m2,
            "ss_tail_variance_m2":
                ss_tail_variance_m2,
            "ss_total_variance_m2":
                ss_total_variance_m2,
            "hm0_ig_m":
                hm0_ig_m,
            "hm0_ss_direct_m":
                hm0_ss_direct_m,
            "hm0_ss_total_m":
                hm0_ss_total_m,
            "candidate_fc_hz":
                candidate_fc_hz,
            "kp_at_candidate_fc":
                kp_at_candidate_fc,
            "gain_at_candidate_fc":
                gain_at_candidate_fc,
            "candidate_fc_safe":
                candidate_fc_safe,
            "tail_level_at_fc_m2_hz":
                tail_level_at_fc_m2_hz,
            "ig_to_ss_total_variance_ratio":
                ig_to_ss_total_variance_ratio,
        }
    )

    # --------------------------------------------------------
    # Save spectrum
    # --------------------------------------------------------

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
                "eta_psd_tail_m2_hz":
                    eta_psd_tail,
                "eta_psd_extended_m2_hz":
                    eta_psd_extended,
                "pressure_response_factor":
                    kp,
                "pressure_amplitude_gain":
                    amplitude_gain,
                "candidate_fc_hz":
                    candidate_fc_hz,
                "kp_at_candidate_fc":
                    kp_at_candidate_fc,
                "gain_at_candidate_fc":
                    gain_at_candidate_fc,
                "candidate_fc_safe":
                    candidate_fc_safe,
                "tail_level_at_fc_m2_hz":
                    tail_level_at_fc_m2_hz,
                "wavenumber_rad_m":
                    wavenumber,
            }
        )
    )

# ============================================================
# BUILD TIME-RESOLVED ADV PRESSURE TABLE
# ============================================================

directional_pressure = (
    pd.concat(directional_pressure_records, ignore_index=True)
    if directional_pressure_records
    else pd.DataFrame()
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

# spectra_file = (
#     output_folder
#     / f"{case_id}_surface_elevation_spectra.pkl"
# )

# One shared, long-format table for comparison across all cases.
# Re-running a case replaces that case's existing rows rather than duplicating
# them.
common_wave_heights_file = (
    output_folder
    / "all_cases_wave_heights_common_fc.csv"
)

statistics.to_csv(statistics_file, index=False)
# spectra.to_pickle(spectra_file)

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
    "ss_high_hz",
    "candidate_fc_hz",
    "kp_at_candidate_fc",
    "gain_at_candidate_fc",
    "candidate_fc_safe",
    "hm0_ig_m",
    "hm0_ss_direct_m",
    "hm0_ss_total_m",
    "ss_direct_variance_m2",
    "ss_tail_variance_m2",
    "ss_total_variance_m2",
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
comparison["ss_high_hz"] = ss_high_hz
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

print("\nSaved spectral statistics:")
print(statistics_file)

# print("\nSaved spectra:")
# print(spectra_file)

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

# # ============================================================
# # SAVE ADCP PRESSURE FOR PRESSURE-VELOCITY ANALYSIS
# # SHARED FILE FOR F1 / F3 ADCP PRESSURE
# # ============================================================

# directional_pressure_file = (
#     output_folder
#     / "all_cases_ADCP_directional_pressure.parquet"
# )

# if directional_pressure.empty:
#     print(
#         "\nNo accepted ADV pressure time series "
#         "available for directional analysis."
#     )
# else:
#     if directional_pressure_file.exists():
#         old = pd.read_parquet(directional_pressure_file)
#         if "case_id" not in old.columns:
#             raise KeyError(
#                 "Existing ADV directional-pressure file "
#                 "does not contain case_id."
#             )
#         old = old.loc[old["case_id"] != case_id]
#         directional_pressure = pd.concat(
#             [old, directional_pressure],
#             ignore_index=True,
#         )

#     directional_pressure = (
#         directional_pressure
#         .sort_values(["case_id", "time", "analysis_block_number"])
#         .reset_index(drop=True)
#     )
#     directional_pressure.to_parquet(
#         directional_pressure_file,
#         index=False,
#     )
#     print(
#         "\nSaved ADV directional pressure time series:\n"
#         f"{directional_pressure_file}"
#     )
#     print(directional_pressure.groupby("case_id").size())

# ============================================================
# SAVE BURST-LEVEL ADCP PRESSURE / WAVE STATISTICS
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
    "pressure_std_pa",
    "pressure_frozen_fraction",
    "ig_variance_m2",
    "ss_direct_variance_m2",
    "ss_tail_variance_m2",
    "ss_total_variance_m2",
    "hm0_ig_m",
    "hm0_ss_direct_m",
    "hm0_ss_total_m",
    "candidate_fc_hz",
    "kp_at_candidate_fc",
    "gain_at_candidate_fc",
    "candidate_fc_safe",
    "tail_level_at_fc_m2_hz",
    "ig_to_ss_total_variance_ratio",
]

pressure_save = accepted[pressure_save_columns].copy()
pressure_save.insert(0, "case_name", case_label)
pressure_save.insert(0, "case_id", case_id)

# Save processing settings for traceability.
pressure_save["sensor_height_above_bed_m"] = z_pressure_m
pressure_save["pressure_sampling_frequency_hz"] = fs
pressure_save["pressure_sampling_interval_s"] = regular_dt_seconds
pressure_save["ig_low_hz"] = ig_low_hz
pressure_save["ig_high_hz"] = ig_high_hz
pressure_save["ss_low_hz"] = ss_low_hz
pressure_save["ss_high_hz"] = ss_high_hz
pressure_save["candidate_fc_hz"] = candidate_fc_hz
pressure_save["tail_exponent"] = tail_exponent
pressure_save["tail_anchor_bins"] = tail_anchor_bins
pressure_save["detrend_order"] = detrend_order
pressure_save["welch_segment_seconds"] = welch_segment_seconds
pressure_save["overlap_fraction"] = overlap_fraction
pressure_save["max_pressure_amplitude_gain"] = max_pressure_amplitude_gain

if pressure_statistics_file.exists():
    old = pd.read_csv(
        pressure_statistics_file,
        parse_dates=["start_time", "end_time", "mid_time"],
    )
    if "case_id" not in old.columns:
        raise KeyError(
            "Existing ADCP pressure-statistics file "
            "does not contain case_id."
        )
    old = old.loc[old["case_id"] != case_id]
    pressure_save = pd.concat(
        [old, pressure_save],
        ignore_index=True,
    )

pressure_save = (
    pressure_save
    .sort_values(["case_id", "mid_time", "analysis_block_number"])
    .reset_index(drop=True)
)
pressure_save.to_csv(
    pressure_statistics_file,
    index=False,
)

print(
    "\nSaved shared ADV pressure/wave statistics:\n"
    f"{pressure_statistics_file}"
)
print(pressure_save.groupby("case_id").size())

print("\nAccepted-burst statistics:")
print(
    accepted[
        [
            "mean_water_depth_m",
            "kp_at_candidate_fc",
            "gain_at_candidate_fc",
            "candidate_fc_safe",
            "hm0_ig_m",
            "hm0_ss_direct_m",
            "hm0_ss_total_m",
            "ss_direct_variance_m2",
            "ss_tail_variance_m2",
            "ss_total_variance_m2",
            "ig_to_ss_total_variance_ratio",
        ]
    ].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
)

print(
    f"\nFixed pressure cutoff: {candidate_fc_hz:.3f} Hz"
)
print(
    "Candidate cutoff safe bursts:",
    int(accepted["candidate_fc_safe"].sum()),
    "/",
    len(accepted),
)
print(
    "Maximum gain at candidate fc:",
    accepted["gain_at_candidate_fc"].max(),
)
print(
    f"Hm0_SS_direct uses {ss_low_hz:.3f}-{candidate_fc_hz:.3f} Hz "
    "from directly pressure-reconstructed surface elevation."
)
print(
    f"Hm0_SS_total adds an equilibrium f^(-{tail_exponent:g}) tail from "
    f"{candidate_fc_hz:.3f} to {ss_high_hz:.3f} Hz."
)

dt = accepted["mid_time"].diff().dt.total_seconds()

accepted["hm0_ig_plot"] = accepted["hm0_ig_m"]
accepted["hm0_ss_direct_plot"] = accepted["hm0_ss_direct_m"]
accepted["hm0_ss_total_plot"] = accepted["hm0_ss_total_m"]

plt.figure(figsize=(12, 5))

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ig_plot"],
    label=r"$H_{m0,IG}$",
)

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ss_direct_plot"],
    label=rf"$H_{{m0,SS}}$ direct (< {candidate_fc_hz:.3f} Hz)",
)

plt.plot(
    accepted["mid_time"],
    accepted["hm0_ss_total_plot"],
    label=rf"$H_{{m0,SS}}$ direct + $f^{{-{tail_exponent:g}}}$ tail",
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

    direct_data = selected.loc[
        (selected["frequency_hz"] > 0)
        & selected["eta_psd_m2_hz"].notna()
    ]

    tail_data = selected.loc[
        (selected["frequency_hz"] > 0)
        & selected["eta_psd_tail_m2_hz"].notna()
    ]

    plt.figure(figsize=(9, 6))

    # Direct pressure-derived spectrum
    plt.loglog(
        direct_data["frequency_hz"],
        direct_data["eta_psd_m2_hz"],
        label="Pressure-derived surface elevation",
    )

    # Equilibrium tail
    plt.loglog(
        tail_data["frequency_hz"],
        tail_data["eta_psd_tail_m2_hz"],
        linestyle="--",
        label=rf"$f^{{-{tail_exponent:g}}}$ tail",
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

    # Fixed cutoff frequency
    plt.axvline(
        candidate_fc_hz,
        linestyle="--",
        label=rf"$f_c$ = {candidate_fc_hz:.3f} Hz",
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
            eta_direct_median=(
                "eta_psd_m2_hz",
                "median",
            ),
            eta_direct_q25=(
                "eta_psd_m2_hz",
                lambda x: x.quantile(0.25),
            ),
            eta_direct_q75=(
                "eta_psd_m2_hz",
                lambda x: x.quantile(0.75),
            ),
            eta_tail_median=(
                "eta_psd_tail_m2_hz",
                "median",
            ),
            eta_tail_q25=(
                "eta_psd_tail_m2_hz",
                lambda x: x.quantile(0.25),
            ),
            eta_tail_q75=(
                "eta_psd_tail_m2_hz",
                lambda x: x.quantile(0.75),
            ),
        )
    )

    direct_data = median_spectrum.loc[
        (median_spectrum["frequency_hz"] > 0)
        & median_spectrum["eta_direct_median"].notna()
    ]

    tail_data = median_spectrum.loc[
        (median_spectrum["frequency_hz"] > 0)
        & median_spectrum["eta_tail_median"].notna()
    ]

    plt.figure(figsize=(9, 6))

    # Median direct spectrum
    plt.loglog(
        direct_data["frequency_hz"],
        direct_data["eta_direct_median"],
        label="Median pressure-derived spectrum",
    )

    plt.fill_between(
        direct_data["frequency_hz"],
        direct_data["eta_direct_q25"],
        direct_data["eta_direct_q75"],
        alpha=0.2,
        label="Direct-spectrum IQR",
    )

    # Median equilibrium tail
    plt.loglog(
        tail_data["frequency_hz"],
        tail_data["eta_tail_median"],
        linestyle="--",
        label=rf"Median $f^{{-{tail_exponent:g}}}$ tail",
    )

    plt.fill_between(
        tail_data["frequency_hz"],
        tail_data["eta_tail_q25"],
        tail_data["eta_tail_q75"],
        alpha=0.15,
        label="Tail IQR",
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

    plt.axvline(
        candidate_fc_hz,
        linestyle="--",
        label=rf"$f_c$ = {candidate_fc_hz:.3f} Hz",
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Surface-elevation PSD (m$^2$/Hz)")
    plt.ylim(1e-3, 6e-1)

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
            eta_direct_mean=(
                "eta_psd_m2_hz",
                "mean",
            ),
            eta_direct_std=(
                "eta_psd_m2_hz",
                "std",
            ),
            eta_tail_mean=(
                "eta_psd_tail_m2_hz",
                "mean",
            ),
            eta_tail_std=(
                "eta_psd_tail_m2_hz",
                "std",
            ),
        )
    )

    direct_data = mean_spectrum.loc[
        (mean_spectrum["frequency_hz"] > 0)
        & mean_spectrum["eta_direct_mean"].notna()
    ]

    tail_data = mean_spectrum.loc[
        (mean_spectrum["frequency_hz"] > 0)
        & mean_spectrum["eta_tail_mean"].notna()
    ]

    plt.figure(figsize=(9, 6))

    # Mean direct spectrum
    plt.loglog(
        direct_data["frequency_hz"],
        direct_data["eta_direct_mean"],
        label="Mean pressure-derived spectrum",
    )

    # Mean equilibrium tail
    plt.loglog(
        tail_data["frequency_hz"],
        tail_data["eta_tail_mean"],
        linestyle="--",
        label=rf"Mean $f^{{-{tail_exponent:g}}}$ tail",
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

    plt.axvline(
        candidate_fc_hz,
        linestyle="--",
        label=rf"$f_c$ = {candidate_fc_hz:.3f} Hz",
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Surface-elevation PSD (m$^2$/Hz)")
    plt.ylim(1e-3, 6e-1)

    plt.title(
        f"Mean surface-elevation spectrum ({case_label})"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()
    
# ============================================================
# TIME-FREQUENCY HEATMAP OF SURFACE-ELEVATION SPECTRUM
# ============================================================

if not spectra.empty:

    heatmap = spectra.copy()

    heatmap["mid_time"] = (
        pd.to_datetime(heatmap["start_time"])
        + (
            pd.to_datetime(heatmap["end_time"])
            - pd.to_datetime(heatmap["start_time"])
        ) / 2
    )

    fmin_plot = 0.005
    fmax_plot = ss_high_hz

    heatmap = heatmap.loc[
        (heatmap["frequency_hz"] >= fmin_plot)
        & (heatmap["frequency_hz"] <= fmax_plot)
        & heatmap["eta_psd_extended_m2_hz"].notna()
    ].copy()

    if not heatmap.empty:

        # ----------------------------------------------------
        # Frequency x time matrix
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Put spectra on a regular grid using the ACTUAL ADCP burst cadence.
        # For the present ADCP deployment this is typically ~1 hour.
        # Missing bursts remain NaN, so true deployment gaps stay white.
        # ----------------------------------------------------
        spectrum_matrix = heatmap.pivot_table(
            index="frequency_hz",
            columns="mid_time",
            values="eta_psd_extended_m2_hz",
            aggfunc="mean",
        ).sort_index(axis=1)

        frequency = spectrum_matrix.index.to_numpy(dtype=float)
        actual_time_raw = pd.DatetimeIndex(spectrum_matrix.columns).sort_values()

        spacing_s = pd.Series(actual_time_raw).diff().dt.total_seconds()
        spacing_s = spacing_s[np.isfinite(spacing_s) & (spacing_s > 0)]
        nominal_spacing_s = float(spacing_s.median()) if len(spacing_s) else 3600.0
        nominal_spacing_s = max(1.0, round(nominal_spacing_s))
        grid_freq = pd.to_timedelta(nominal_spacing_s, unit="s")

        # Snap burst centers to the regular cadence relative to the first burst.
        t0 = actual_time_raw.min()
        step_number = np.rint(
            (actual_time_raw - t0).total_seconds() / nominal_spacing_s
        ).astype(int)
        actual_time = pd.DatetimeIndex(t0 + step_number * grid_freq)
        spectrum_matrix.columns = actual_time

        time_grid = pd.date_range(
            start=actual_time.min(),
            end=actual_time.max(),
            freq=grid_freq,
        )
        
        # Insert all missing nominal ADCP burst times as NaN
        spectrum_matrix = spectrum_matrix.reindex(
            columns=time_grid
        )
        
        S_plot = spectrum_matrix.to_numpy(dtype=float)
        
        # ----------------------------------------------------
        # Log10 PSD
        # ----------------------------------------------------
        
        log_S_eta = np.full_like(
            S_plot,
            np.nan,
            dtype=float,
        )
        
        valid = np.isfinite(S_plot) & (S_plot > 0)
        
        log_S_eta[valid] = np.log10(
            S_plot[valid]
        )
        
        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        mesh = ax.pcolormesh(
            time_grid,
            frequency,
            np.ma.masked_invalid(log_S_eta),
            shading="auto",
        )
        
        cbar = fig.colorbar(mesh, ax=ax)
        
        cbar.set_label(
            r"$\log_{10} S_{\eta\eta}$ (m$^2$/Hz)"
        )
        
        ax.axhline(
            ig_high_hz,
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Current IG/SS boundary "
                f"({ig_high_hz:.3f} Hz)"
            ),
        )
        
        ax.set_yscale("log")
        ax.set_ylim(fmin_plot, fmax_plot)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Hz)")
        
        ax.set_title(
            f"{case_label}: time-frequency evolution "
            "of surface-elevation spectrum"
        )
        
        ax.legend()
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()