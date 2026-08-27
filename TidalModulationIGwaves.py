# -*- coding: utf-8 -*-
"""
ADV-pressure + ADCP-velocity IG cross-spectral analysis.

Purpose
-------
Use time-resolved ADV pressure together with ADCP horizontal velocity
to test whether IG-band velocity fluctuations are coherent with the
surface-wave pressure signal.

For each overlapping ADV-pressure / ADCP-velocity burst and each ADCP cell:
1. Match by frame (F1/F3), not by full case_id.
2. Interpolate ADCP velocity to the ADV-pressure timestamps.
3. Compute the IG-band (0.005-0.05 Hz) horizontal principal velocity axis.
4. Compute pressure-velocity cross-spectrum, coherence and phase for:
       - original cross-shore velocity U
       - original alongshore velocity V
       - IG principal-axis velocity Ua
5. Save burst-level diagnostics and full frequency-dependent spectra.

Inputs
------
all_cases_ADCP_current_wave_axis.csv
all_cases_ADV_pressure_wave_statistics.csv
all_cases_ADCP_directional_velocity.parquet
all_cases_ADV_directional_pressure.parquet
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch
import pyarrow.dataset as ds


# ============================================================
# USER SETTINGS
# ============================================================

input_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)

current_file = (
    input_folder
    / "all_cases_ADCP_current_wave_axis.csv"
)

pressure_statistics_file = (
    input_folder
    / "all_cases_ADV_pressure_wave_statistics.csv"
)

velocity_file = (
    input_folder
    / "all_cases_ADCP_directional_velocity.parquet"
)

pressure_file = (
    input_folder
    / "all_cases_ADV_directional_pressure.parquet"
)

output_folder = input_folder

burst_output_file = (
    output_folder
    / "all_cases_ADVpressure_ADCPvelocity_IG_diagnostics.csv"
)

spectral_output_file = (
    output_folder
    / "all_cases_ADVpressure_ADCPvelocity_cross_spectra.parquet"
)

# IG band
ig_low_hz = 0.005
ig_high_hz = 0.050

# Welch settings
welch_segment_seconds = 512.0
overlap_fraction = 0.5

# Matching / QC
minimum_overlap_fraction = 0.95
minimum_samples = 300
statistics_merge_tolerance = pd.Timedelta("20min")


# ============================================================
# FUNCTIONS
# ============================================================

def extract_frame_id(values):
    """
    Extract frame identifier F1, F3, etc. from case IDs.

    Examples
    --------
    DVN_F3_ADV05 -> F3
    DVN_F3_ADCP -> F3
    DVN_F3_ADCP-HR -> F3
    """
    s = values.astype(str)
    frame = s.str.extract(r"(F\d+)", expand=False)
    return frame.str.upper()


def ensure_ns_utc(series):
    """Force timestamps to datetime64[ns, UTC]."""
    return pd.to_datetime(
        series, utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")


def detect_fs(time):
    """Estimate regular sampling frequency from timestamps."""
    dt = (
        pd.Series(time)
        .diff()
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        return np.nan

    cutoff = np.nanpercentile(dt, 25)
    regular = dt[dt <= 1.5 * cutoff]

    if regular.size == 0:
        regular = dt

    median_dt = float(np.nanmedian(regular))
    return 1.0 / median_dt if median_dt > 0 else np.nan


def spectral_settings(fs, n):
    """Return Welch nperseg and noverlap."""
    nperseg = min(
        int(round(welch_segment_seconds * fs)),
        n,
    )

    if nperseg < 16:
        raise ValueError("Too few samples for Welch analysis.")

    noverlap = min(
        int(round(overlap_fraction * nperseg)),
        nperseg - 1,
    )

    return nperseg, noverlap


def calculate_velocity_spectra(u, v, fs):
    """Horizontal velocity auto-spectra and U-V cross-spectrum."""
    nperseg, noverlap = spectral_settings(fs, len(u))

    kwargs = dict(
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    f, suu = welch(u, **kwargs)
    fv, svv = welch(v, **kwargs)
    fuv, suv = csd(u, v, **kwargs)

    if not (
        np.allclose(f, fv)
        and np.allclose(f, fuv)
    ):
        raise RuntimeError(
            "Velocity spectral frequency grids do not match."
        )

    return f, suu, svv, suv


def calculate_ig_principal_axis(
    frequency,
    suu,#u spectrum
    svv,#v spectrum
    suv,#cross spetrum
    fmin,
    fmax,
):
    """
    Principal horizontal velocity axis from IG-band spectral
    variance/covariance only.

    Angle convention:
        0 deg   = +cross-shore axis
        90 deg  = +alongshore axis

    Because this is an axis, theta and theta+180 deg are equivalent.
    """
    band = (
        (frequency >= fmin)
        & (frequency < fmax)
        & np.isfinite(suu)
        & np.isfinite(svv)
        & np.isfinite(suv)
    )

    if band.sum() < 2:
        return (np.nan,) * 8

    f = frequency[band]

    var_u = float(
        np.trapezoid(suu[band], f)
    )
    var_v = float(
        np.trapezoid(svv[band], f)
    )
    cov_uv = float(
        np.trapezoid(
            np.real(suv[band]),
            f,
        )
    )
    
    # eigenvector of the covariance matrix associated with the largest eigenvalue
    # 0.5*tan2^{-1}(2.0 * cov_uv, var_u - var_v)
    theta_rad = 0.5 * np.arctan2(
        2.0 * cov_uv,
        var_u - var_v,
    )
    theta_deg = float(
        np.degrees(theta_rad) % 180.0
    )

    discriminant = np.sqrt(
        (var_u - var_v) ** 2
        + 4.0 * cov_uv ** 2
    )

    major_variance = float(
        0.5 * (
            var_u
            + var_v
            + discriminant
        )
    )
    minor_variance = float(
        0.5 * (
            var_u
            + var_v
            - discriminant
        )
    )

    axis_variance_ratio = (
        major_variance / minor_variance
        if (
            np.isfinite(minor_variance)
            and minor_variance > 0
        )
        else np.nan
    )

    return (
        theta_rad,
        theta_deg,
        var_u,
        var_v,
        cov_uv,
        major_variance,
        minor_variance,
        axis_variance_ratio,
    )


def calculate_pressure_velocity_spectra(
    pressure,
    u,
    v,
    axis_velocity,
    horizontal_velocity,
    fs,
):
    """
    Calculate pressure/velocity PSDs, cross-spectra and coherence for:
        - cross-shore velocity U
        - alongshore velocity V
        - principal-axis velocity Ua
        - horizontal velocity magnitude Uh = sqrt(U^2 + V^2)
    """

    nperseg, noverlap = spectral_settings(
        fs,
        len(pressure),
    )

    psd_kwargs = dict(
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    coherence_kwargs = dict(
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )

    # --------------------------------------------------------
    # Autospectra
    # --------------------------------------------------------

    frequency, spp = welch(
        pressure,
        **psd_kwargs,
    )

    _, suu = welch(
        u,
        **psd_kwargs,
    )

    _, svv = welch(
        v,
        **psd_kwargs,
    )

    _, saa = welch(
        axis_velocity,
        **psd_kwargs,
    )

    _, shh = welch(
        horizontal_velocity,
        **psd_kwargs,
    )

    # --------------------------------------------------------
    # Pressure-velocity cross-spectra
    # --------------------------------------------------------

    _, spu = csd(
        pressure,
        u,
        **psd_kwargs,
    )

    _, spv = csd(
        pressure,
        v,
        **psd_kwargs,
    )

    _, spa = csd(
        pressure,
        axis_velocity,
        **psd_kwargs,
    )

    _, sph = csd(
        pressure,
        horizontal_velocity,
        **psd_kwargs,
    )

    # --------------------------------------------------------
    # Magnitude-squared coherence
    # --------------------------------------------------------

    _, cpu = coherence(
        pressure,
        u,
        **coherence_kwargs,
    )

    _, cpv = coherence(
        pressure,
        v,
        **coherence_kwargs,
    )

    _, cpa = coherence(
        pressure,
        axis_velocity,
        **coherence_kwargs,
    )

    _, cph = coherence(
        pressure,
        horizontal_velocity,
        **coherence_kwargs,
    )

    return (
        frequency,
        spp,
        suu,
        svv,
        saa,
        shh,
        spu,
        spv,
        spa,
        sph,
        cpu,
        cpv,
        cpa,
        cph,
    )


def circular_weighted_phase(
    phase_rad,
    weights,
):
    """Weighted circular mean phase."""
    phase_rad = np.asarray(
        phase_rad,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        np.isfinite(phase_rad)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if valid.sum() == 0:
        return np.nan

    vector = np.sum(
        weights[valid]
        * np.exp(
            1j * phase_rad[valid]
        )
    )

    if np.abs(vector) == 0:
        return np.nan

    return np.angle(vector)


def band_summary(
    frequency,
    coherence_values,
    cross_spectrum,
    pressure_psd,
    fmin,
    fmax,
):
    """
    Summarize pressure-velocity relationship over one frequency band.

    Returns
    -------
    median_coherence
    pressure-energy-weighted coherence
    cross-spectral-amplitude-weighted phase [deg]
    number of frequency bins
    """
    band = (
        (frequency >= fmin)
        & (frequency < fmax)
        & np.isfinite(coherence_values)
        & np.isfinite(pressure_psd)
        & np.isfinite(cross_spectrum)
    )

    number_of_bins = int(
        band.sum()
    )

    if number_of_bins < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            number_of_bins,
        )

    coh = coherence_values[band]
    ppsd = pressure_psd[band]
    cross = cross_spectrum[band]

    median_coherence = float(
        np.nanmedian(coh)
    )

    pressure_sum = float(
        np.nansum(ppsd)
    )

    weighted_coherence = (
        float(
            np.nansum(coh * ppsd)
            / pressure_sum
        )
        if pressure_sum > 0
        else np.nan
    )

    phase_rad = circular_weighted_phase(
        np.angle(cross),
        np.abs(cross),
    )

    phase_deg = (
        float(np.degrees(phase_rad))
        if np.isfinite(phase_rad)
        else np.nan
    )

    return (
        median_coherence,
        weighted_coherence,
        phase_deg,
        number_of_bins,
    )


# ============================================================
# READ SMALL BURST-LEVEL CSV FILES
# ============================================================

print("Reading ADCP current / velocity statistics:")
print(current_file)
current_stats = pd.read_csv(current_file)

print("\nReading ADV pressure statistics:")
print(pressure_statistics_file)
pressure_stats = pd.read_csv(pressure_statistics_file)


# ============================================================
# CASE IDs + FRAME IDs FROM SMALL CSV FILES
# ============================================================

for name, df in [
    ("current_stats", current_stats),
    ("pressure_stats", pressure_stats),
]:
    if "case_id" not in df.columns:
        raise KeyError(f"{name} does not contain 'case_id'.")

    df["case_id"] = df["case_id"].astype(str)
    df["frame_id"] = extract_frame_id(df["case_id"])

    if df["frame_id"].isna().any():
        bad = (
            df.loc[df["frame_id"].isna(), "case_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Could not extract frame ID from {name}: {bad}"
        )


# ------------------------------------------------------------
# Normalize CSV timestamps
# ------------------------------------------------------------

current_stats["mid_time"] = ensure_ns_utc(
    current_stats["mid_time"]
)
pressure_stats["mid_time"] = ensure_ns_utc(
    pressure_stats["mid_time"]
)

for df in [current_stats, pressure_stats]:
    if "start_time" in df.columns:
        df["start_time"] = ensure_ns_utc(df["start_time"])
    if "end_time" in df.columns:
        df["end_time"] = ensure_ns_utc(df["end_time"])


# ============================================================
# OPEN LARGE PARQUETS AS PYARROW DATASETS
# DO NOT LOAD THEM INTO PANDAS
# ============================================================

print("\nOpening ADCP directional velocity dataset:")
print(velocity_file)
velocity_dataset = ds.dataset(
    velocity_file,
    format="parquet",
)

print("\nOpening ADV directional pressure dataset:")
print(pressure_file)
pressure_dataset = ds.dataset(
    pressure_file,
    format="parquet",
)


# ============================================================
# CHECK PARQUET SCHEMAS
# ============================================================

required_velocity_columns = {
    "case_id",
    "cell_number",
    "height_relative_to_frame_bottom_m",
    "time",
    "cross_velocity_detrended_m_s",
    "along_velocity_detrended_m_s",
}

required_pressure_columns = {
    "case_id",
    "analysis_block_number",
    "time",
    "pressure_head_detrended_m",
    "mean_water_depth_m",
}

velocity_columns = set(velocity_dataset.schema.names)
pressure_columns = set(pressure_dataset.schema.names)

missing_velocity = required_velocity_columns - velocity_columns
missing_pressure = required_pressure_columns - pressure_columns

if missing_velocity:
    raise KeyError(
        "Missing ADCP velocity columns: "
        f"{sorted(missing_velocity)}"
    )

if missing_pressure:
    raise KeyError(
        "Missing ADV pressure columns: "
        f"{sorted(missing_pressure)}"
    )


# ============================================================
# CASE LOOKUPS FROM SMALL CSV FILES
# ============================================================

velocity_case_lookup = (
    current_stats[["frame_id", "case_id"]]
    .drop_duplicates()
    .groupby("frame_id")["case_id"]
    .apply(list)
    .to_dict()
)

pressure_case_lookup = (
    pressure_stats[["frame_id", "case_id"]]
    .drop_duplicates()
    .groupby("frame_id")["case_id"]
    .apply(list)
    .to_dict()
)

print("\nADCP velocity cases by frame:")
print(velocity_case_lookup)

print("\nADV pressure cases by frame:")
print(pressure_case_lookup)


# ============================================================
# AVAILABLE FRAMES
# ============================================================

common_frames = sorted(
    set(velocity_case_lookup)
    & set(pressure_case_lookup)
)

if not common_frames:
    raise RuntimeError(
        "No common frames between ADV pressure and ADCP velocity."
    )

print("\nCommon ADV-pressure / ADCP-velocity frames:")
print(common_frames)


# ============================================================
# PRESSURE BURST INDEX
# Use the SMALL pressure-statistics CSV to identify bursts.
# ============================================================

required_pressure_stats = {
    "case_id",
    "frame_id",
    "analysis_block_number",
    "start_time",
    "end_time",
    "mid_time",
}

missing_pressure_stats = (
    required_pressure_stats
    - set(pressure_stats.columns)
)

if missing_pressure_stats:
    raise KeyError(
        "Pressure statistics CSV is missing columns: "
        f"{sorted(missing_pressure_stats)}"
    )

pressure_bursts = (
    pressure_stats.loc[
        pressure_stats["frame_id"].isin(common_frames)
    ]
    .dropna(
        subset=[
            "case_id",
            "analysis_block_number",
            "start_time",
            "end_time",
        ]
    )
    .drop_duplicates(
        subset=[
            "case_id",
            "analysis_block_number",
        ]
    )
    .sort_values(
        [
            "frame_id",
            "case_id",
            "start_time",
        ]
    )
    .reset_index(drop=True)
)

print("\nPressure bursts available for matching:")
print(
    pressure_bursts.groupby(
        ["frame_id", "case_id"]
    ).size()
)


# ============================================================
# PRESSURE-VELOCITY CROSS-SPECTRAL ANALYSIS
# ============================================================

burst_records = []
spectral_records = []

pressure_columns_to_read = [
    "case_id",
    "analysis_block_number",
    "time",
    "pressure_head_detrended_m",
    "mean_water_depth_m",
]

for _, burst_meta in pressure_bursts.iterrows():
    frame_id = burst_meta["frame_id"]
    pressure_case_id = str(burst_meta["case_id"])
    pressure_block_number = int(
        burst_meta["analysis_block_number"]
    )

    # --------------------------------------------------------
    # Read ONLY this ADV pressure burst from Parquet
    # --------------------------------------------------------

    pressure_filter = (
        (ds.field("case_id") == pressure_case_id)
        & (
            ds.field("analysis_block_number")
            == pressure_block_number
        )
    )

    p_block = (
        pressure_dataset
        .to_table(
            columns=pressure_columns_to_read,
            filter=pressure_filter,
        )
        .to_pandas()
    )

    if p_block.empty:
        continue

    p_block["time"] = ensure_ns_utc(p_block["time"])

    p_block = (
        p_block
        .dropna(
            subset=[
                "time",
                "pressure_head_detrended_m",
            ]
        )
        .sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

    if len(p_block) < minimum_samples:
        continue

    p_start = p_block["time"].iloc[0]
    p_end = p_block["time"].iloc[-1]
    p_mid = p_start + (p_end - p_start) / 2

    fs_pressure = detect_fs(p_block["time"])

    if (
        not np.isfinite(fs_pressure)
        or fs_pressure <= 0
    ):
        continue

    # --------------------------------------------------------
    # Find simultaneous ADCP velocity from the same frame.
    #
    # ADV pressure case IDs such as DVN_F3_ADV05 are NOT
    # expected to equal ADCP velocity case IDs such as
    # DVN_F3_ADCP or DVN_F3_ADCP-HR.
    # --------------------------------------------------------

  # --------------------------------------------------------
    # Read ONLY simultaneous ADCP velocity samples
    # for this pressure burst.
    # --------------------------------------------------------
    
    velocity_cases = velocity_case_lookup.get(
        frame_id,
        [],
    )
    
    if len(velocity_cases) == 0:
        continue
    
    if len(velocity_cases) > 1:
        raise RuntimeError(
            f"More than one ADCP velocity case found for "
            f"{frame_id}: {velocity_cases}"
        )
    
    velocity_case_id = velocity_cases[0]
    
    velocity_columns_to_read = [
        "case_id",
        "cell_number",
        "height_relative_to_frame_bottom_m",
        "time",
        "cross_velocity_detrended_m_s",
        "along_velocity_detrended_m_s",
    ]
    
    # Arrow timestamps must use Python datetime-compatible values.
    p_start_filter = p_start.to_pydatetime()
    p_end_filter = p_end.to_pydatetime()
    
    velocity_filter = (
        (ds.field("case_id") == velocity_case_id)
        & (ds.field("time") >= p_start_filter)
        & (ds.field("time") <= p_end_filter)
    )
    
    v_overlap = (
        velocity_dataset
        .to_table(
            columns=velocity_columns_to_read,
            filter=velocity_filter,
        )
        .to_pandas()
    )

    if v_overlap.empty:
        continue
    
    v_overlap["time"] = ensure_ns_utc(
        v_overlap["time"]
    )
    
    v_overlap = (
        v_overlap
        .sort_values(
            [
                "cell_number",
                "time",
            ]
        )
        .reset_index(drop=True)
    )

    cell_numbers = sorted(
        v_overlap[
            "cell_number"
        ]
        .dropna()
        .unique()
    )

    for cell_number in cell_numbers:
        v_cell = (
            v_overlap.loc[
                v_overlap["cell_number"]
                == cell_number
            ]
            .sort_values("time")
            .drop_duplicates(
                subset="time"
            )
            .copy()
        )

        if len(v_cell) < minimum_samples:
            continue

        height_m = float(
            v_cell[
                "height_relative_to_frame_bottom_m"
            ].iloc[0]
        )

        # ====================================================
        # TIME ARRAYS
        # ====================================================

        p_seconds = (
            p_block["time"]
            .astype("int64")
            .to_numpy(
                dtype=np.float64
            )
            / 1e9
        )

        v_seconds = (
            v_cell["time"]
            .astype("int64")
            .to_numpy(
                dtype=np.float64
            )
            / 1e9
        )

        pressure_head = (
            p_block[
                "pressure_head_detrended_m"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        u_raw = (
            v_cell[
                "cross_velocity_detrended_m_s"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        v_raw = (
            v_cell[
                "along_velocity_detrended_m_s"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        # ====================================================
        # CLEAN ADCP VELOCITY
        # ====================================================

        finite_velocity = (
            np.isfinite(v_seconds)
            & np.isfinite(u_raw)
            & np.isfinite(v_raw)
        )

        if (
            finite_velocity.sum()
            < minimum_samples
        ):
            continue

        v_seconds = (
            v_seconds[
                finite_velocity
            ]
        )
        u_raw = (
            u_raw[
                finite_velocity
            ]
        )
        v_raw = (
            v_raw[
                finite_velocity
            ]
        )


        # ====================================================
        # MATCH PRESSURE TIMES TO ADCP VELOCITY COVERAGE
        # ====================================================

        inside = (
            (
                p_seconds
                >= v_seconds.min()
            )
            & (
                p_seconds
                <= v_seconds.max()
            )
            & np.isfinite(
                pressure_head
            )
        )

        velocity_overlap_fraction = float(
            inside.mean()
        )

        if (
            velocity_overlap_fraction
            < minimum_overlap_fraction
        ):
            continue


        # ====================================================
        # INTERPOLATE ADCP VELOCITY TO ADV-PRESSURE TIMES
        # ====================================================

        p_seconds_use = (
            p_seconds[
                inside
            ]
        )

        pressure_use = (
            pressure_head[
                inside
            ]
        )

        u_use = np.interp(
            p_seconds_use,
            v_seconds,
            u_raw,
        )

        v_use = np.interp(
            p_seconds_use,
            v_seconds,
            v_raw,
        )

        finite = (
            np.isfinite(pressure_use)
            & np.isfinite(u_use)
            & np.isfinite(v_use)
        )

        pressure_use = (
            pressure_use[
                finite
            ]
        )
        u_use = (
            u_use[
                finite
            ]
        )
        v_use = (
            v_use[
                finite
            ]
        )

        if len(
            pressure_use
        ) < minimum_samples:
            continue


        # ====================================================
        # IG PRINCIPAL AXIS FROM ADCP VELOCITY
        # ====================================================

        (
            f_velocity,
            suu_velocity,
            svv_velocity,
            suv_velocity,
        ) = calculate_velocity_spectra(
            u_use,
            v_use,
            fs_pressure,
        )

        (
            theta_rad,
            theta_deg,
            ig_var_u,
            ig_var_v,
            ig_cov_uv,
            ig_major_variance,
            ig_minor_variance,
            ig_axis_variance_ratio,
        ) = calculate_ig_principal_axis(
            f_velocity,
            suu_velocity,
            svv_velocity,
            suv_velocity,
            ig_low_hz,
            ig_high_hz,
        )

        if not np.isfinite(
            theta_rad
        ):
            continue

        velocity_axis = (
            u_use
            * np.cos(theta_rad)
            + v_use
            * np.sin(theta_rad)
        )
        
        horizontal_velocity = np.sqrt(
            u_use**2
            + v_use**2
        )

        # ====================================================
        # PRESSURE-VELOCITY CROSS-SPECTRA
        # ====================================================

        (
            frequency,
            spp,
            suu,
            svv,
            saa,
            shh,
            spu,
            spv,
            spa,
            sph,
            cpu,
            cpv,
            cpa,
            cph,
        ) = calculate_pressure_velocity_spectra(
            pressure_use,
            u_use,
            v_use,
            velocity_axis,
            horizontal_velocity,
            fs_pressure,
        )


        # ====================================================
        # IG-BAND COHERENCE + PHASE
        # 0.005-0.05 Hz ONLY
        # ====================================================

        (
            cross_median,
            cross_weighted,
            cross_phase,
            cross_bins,
        ) = band_summary(
            frequency,
            cpu,
            spu,
            spp,
            ig_low_hz,
            ig_high_hz,
        )

        (
            along_median,
            along_weighted,
            along_phase,
            along_bins,
        ) = band_summary(
            frequency,
            cpv,
            spv,
            spp,
            ig_low_hz,
            ig_high_hz,
        )

        (
            axis_median,
            axis_weighted,
            axis_phase,
            axis_bins,
        ) = band_summary(
            frequency,
            cpa,
            spa,
            spp,
            ig_low_hz,
            ig_high_hz,
        )
            
        (
            horizontal_median,
            horizontal_weighted,
            horizontal_phase,
            horizontal_bins,
        ) = band_summary(
            frequency,
            cph,
            sph,
            spp,
            ig_low_hz,
            ig_high_hz,
        )

        # ====================================================
        # BURST-LEVEL RESULT
        # ====================================================

        burst_records.append({
            "frame_id":
                frame_id,

            "pressure_case_id":
                pressure_case_id,

            "velocity_case_id":
                velocity_case_id,

            "pressure_analysis_block_number":
                pressure_block_number,

            "cell_number":
                cell_number,

            "height_relative_to_frame_bottom_m":
                height_m,

            "start_time":
                p_start,

            "end_time":
                p_end,

            "mid_time":
                p_mid,

            "pressure_sampling_frequency_hz":
                fs_pressure,

            "matched_sample_count":
                len(pressure_use),

            "velocity_overlap_fraction":
                velocity_overlap_fraction,

            "mean_water_depth_m":
                float(
                    p_block[
                        "mean_water_depth_m"
                    ].iloc[0]
                ),

            "ig_low_hz":
                ig_low_hz,

            "ig_high_hz":
                ig_high_hz,

            # Principal-axis geometry
            "ig_principal_axis_deg":
                theta_deg,

            "ig_cross_variance_m2_s2":
                ig_var_u,

            "ig_along_variance_m2_s2":
                ig_var_v,

            "ig_uv_covariance_m2_s2":
                ig_cov_uv,

            "ig_major_axis_variance_m2_s2":
                ig_major_variance,

            "ig_minor_axis_variance_m2_s2":
                ig_minor_variance,

            "ig_axis_variance_ratio":
                ig_axis_variance_ratio,

            # Cross-shore pressure-velocity relation
            "ig_coherence_cross_median":
                cross_median,

            "ig_coherence_cross_weighted":
                cross_weighted,

            "ig_phase_cross_deg":
                cross_phase,

            "ig_cross_frequency_bins":
                cross_bins,

            # Alongshore pressure-velocity relation
            "ig_coherence_along_median":
                along_median,

            "ig_coherence_along_weighted":
                along_weighted,

            "ig_phase_along_deg":
                along_phase,

            "ig_along_frequency_bins":
                along_bins,

            # Principal-axis pressure-velocity relation
            "ig_coherence_axis_median":
                axis_median,

            "ig_coherence_axis_weighted":
                axis_weighted,

            "ig_phase_axis_deg":
                axis_phase,

            "ig_axis_frequency_bins":
                axis_bins,
                
            # Horizontal-magnitude pressure-velocity relation
            "ig_coherence_horizontal_median":
                horizontal_median,
            
            "ig_coherence_horizontal_weighted":
                horizontal_weighted,
            
            "ig_phase_horizontal_deg":
                horizontal_phase,
            
            "ig_horizontal_frequency_bins":
                horizontal_bins,
        })


        # ====================================================
        # FULL FREQUENCY-BY-FREQUENCY RESULT
        # ====================================================

        spectral_records.append(
            pd.DataFrame({
                "frame_id":
                    frame_id,

                "pressure_case_id":
                    pressure_case_id,

                "velocity_case_id":
                    velocity_case_id,

                "pressure_analysis_block_number":
                    pressure_block_number,

                "cell_number":
                    cell_number,

                "height_relative_to_frame_bottom_m":
                    height_m,

                "start_time":
                    p_start,

                "end_time":
                    p_end,

                "mid_time":
                    p_mid,

                "ig_principal_axis_deg":
                    theta_deg,

                "frequency_hz":
                    frequency,

                "pressure_head_psd_m2_hz":
                    spp,

                "cross_velocity_psd_m2_s2_hz":
                    suu,

                "along_velocity_psd_m2_s2_hz":
                    svv,

                "axis_velocity_psd_m2_s2_hz":
                    saa,

                "pressure_cross_velocity_csd_real":
                    np.real(spu),

                "pressure_cross_velocity_csd_imag":
                    np.imag(spu),

                "pressure_along_velocity_csd_real":
                    np.real(spv),

                "pressure_along_velocity_csd_imag":
                    np.imag(spv),

                "pressure_axis_velocity_csd_real":
                    np.real(spa),

                "pressure_axis_velocity_csd_imag":
                    np.imag(spa),
                
                "pressure_horizontal_velocity_csd_real":
                    np.real(sph),
                    
                "pressure_horizontal_velocity_csd_imag":
                    np.imag(sph),

                "pressure_cross_velocity_coherence":
                    cpu,

                "pressure_along_velocity_coherence":
                    cpv,

                "pressure_axis_velocity_coherence":
                    cpa,
                    
                "horizontal_velocity_psd_m2_s2_hz":
                    shh,
                
                "pressure_horizontal_velocity_coherence":
                    cph,

                "pressure_cross_velocity_phase_deg":
                    np.degrees(
                        np.angle(spu)
                    ),

                "pressure_along_velocity_phase_deg":
                    np.degrees(
                        np.angle(spv)
                    ),

                "pressure_axis_velocity_phase_deg":
                    np.degrees(
                        np.angle(spa)
                    ),
                    
                "pressure_horizontal_velocity_phase_deg":
                    np.degrees(
                        np.angle(sph)
                    ),
            })
        )


# ============================================================
# BUILD RESULTS
# ============================================================

diagnostics = pd.DataFrame(
    burst_records
)

cross_spectra = (
    pd.concat(
        spectral_records,
        ignore_index=True,
    )
    if spectral_records
    else pd.DataFrame()
)

if diagnostics.empty:
    raise RuntimeError(
        "No overlapping ADV-pressure / ADCP-velocity "
        "bursts passed the matching criteria."
    )

for column in [
    "start_time",
    "end_time",
    "mid_time",
]:
    diagnostics[column] = ensure_ns_utc(
        diagnostics[column]
    )


# ============================================================
# ATTACH ADV PRESSURE-DERIVED WAVE STATISTICS
# ============================================================

pressure_keep = [
    "case_id",
    "frame_id",
    "mid_time",
    "mean_water_depth_m",
    "ig_variance_m2",
    "ss_variance_m2",
    "ss_common_fc_variance_m2",
    "hm0_ig_m",
    "hm0_ss_m",
    "hm0_ss_common_fc_m",
    "ig_to_ss_variance_ratio",
    "ig_to_ss_common_fc_variance_ratio",
]

pressure_keep = [
    column
    for column in pressure_keep
    if column in pressure_stats.columns
]

pressure_merge = (
    pressure_stats[
        pressure_keep
    ]
    .dropna(
        subset=["mid_time"]
    )
    .copy()
)

pressure_merge["mid_time"] = ensure_ns_utc(
    pressure_merge["mid_time"]
)

# Rename case ID so it can be matched explicitly to ADV pressure.
pressure_merge = pressure_merge.rename(
    columns={
        "case_id":
            "pressure_case_id"
    }
)

pressure_merge = (
    pressure_merge
    .sort_values("mid_time")
)

diagnostics = (
    diagnostics
    .sort_values("mid_time")
)

diagnostics = pd.merge_asof(
    diagnostics,
    pressure_merge,
    on="mid_time",
    by=[
        "frame_id",
        "pressure_case_id",
    ],
    direction="nearest",
    tolerance=
        statistics_merge_tolerance,
    suffixes=(
        "",
        "_pressure_stats",
    ),
)


# ============================================================
# ATTACH ADCP CURRENT / VELOCITY-RMS STATISTICS
# ============================================================

current_keep = [
    "case_id",
    "frame_id",
    "mid_time",
    "cell_number",
    "depth_avg_cross_current_m_s",
    "depth_avg_along_current_m_s",
    "depth_avg_current_speed_m_s",
    "depth_avg_current_direction_deg",
    "cross_ig_rms_m_s",
    "along_ig_rms_m_s",
    "horizontal_ig_rms_m_s",
    "cross_ss_rms_m_s",
    "along_ss_rms_m_s",
    "horizontal_ss_rms_m_s",
]

current_keep = [
    column
    for column in current_keep
    if column in current_stats.columns
]

current_merge = (
    current_stats[
        current_keep
    ]
    .dropna(
        subset=["mid_time"]
    )
    .copy()
)

current_merge["mid_time"] = ensure_ns_utc(
    current_merge["mid_time"]
)

current_merge = current_merge.rename(
    columns={
        "case_id":
            "velocity_case_id"
    }
)

current_merge = (
    current_merge
    .sort_values("mid_time")
)

diagnostics["mid_time"] = ensure_ns_utc(
    diagnostics["mid_time"]
)

diagnostics = (
    diagnostics
    .sort_values("mid_time")
)

diagnostics = pd.merge_asof(
    diagnostics,
    current_merge,
    on="mid_time",
    by=[
        "frame_id",
        "velocity_case_id",
        "cell_number",
    ],
    direction="nearest",
    tolerance=
        statistics_merge_tolerance,
    suffixes=(
        "",
        "_velocity_stats",
    ),
)


# ============================================================
# SAVE RESULTS
# ============================================================

diagnostics = (
    diagnostics
    .sort_values(
        [
            "frame_id",
            "mid_time",
            "cell_number",
        ]
    )
    .reset_index(drop=True)
)

diagnostics.to_csv(
    burst_output_file,
    index=False,
)

if not cross_spectra.empty:
    cross_spectra.to_parquet(
        spectral_output_file,
        index=False,
    )

print(
    "\nSaved burst-level diagnostics:"
)
print(
    burst_output_file
)

print(
    "\nSaved full pressure-velocity spectra:"
)
print(
    spectral_output_file
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nMatched ADV-pressure / ADCP-velocity burst counts:"
)

print(
    diagnostics.groupby(
        [
            "frame_id",
            "pressure_case_id",
            "velocity_case_id",
            "cell_number",
        ]
    ).size()
)

summary_columns = [
    "ig_axis_variance_ratio",
    "ig_coherence_cross_weighted",
    "ig_coherence_along_weighted",
    "ig_coherence_axis_weighted",
    "ig_coherence_horizontal_weighted",
    "ig_phase_cross_deg",
    "ig_phase_along_deg",
    "ig_phase_axis_deg",
    "ig_phase_horizontal_deg",
]

print(
    "\nIG pressure-velocity diagnostics:"
)

print(
    diagnostics[
        summary_columns
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
# FIGURE 1
# CROSS / ALONG / PRINCIPAL-AXIS IG COHERENCE VS TIME
# ============================================================
def plot_with_gaps(ax, time, values, gap_factor=3, **kwargs):
    time = pd.Series(time).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)

    dt = time.diff()
    valid_dt = dt[dt.notna() & (dt > pd.Timedelta(0))]

    if valid_dt.empty:
        return ax.plot(time, values, **kwargs)[0]

    gap_threshold = gap_factor * valid_dt.median()
    segment = (dt > gap_threshold).cumsum()

    first_line = None
    fixed_color = kwargs.pop("color", None)
    label = kwargs.pop("label", None)

    for i, idx in enumerate(segment.groupby(segment).groups.values()):
        line, = ax.plot(
            time.iloc[idx],
            values.iloc[idx],
            label=label if i == 0 else None,
            color=fixed_color,
            **kwargs,
        )

        if first_line is None:
            first_line = line
            fixed_color = line.get_color()

    return first_line
        
frames = (
    diagnostics[
        "frame_id"
    ]
    .dropna()
    .unique()
)

frames = (
    diagnostics["frame_id"]
    .dropna()
    .unique()
)

for frame_id in frames:
    dframe = (
        diagnostics.loc[
            diagnostics["frame_id"] == frame_id
        ]
        .copy()
    )

    cells = sorted(
        dframe["cell_number"]
        .dropna()
        .unique()
    )

    fig, axes = plt.subplots(
        len(cells), 1,
        figsize=(12, 2.7 * len(cells)),
        sharex=True,
    )

    if len(cells) == 1:
        axes = [axes]

    for ax, cell in zip(axes, cells):
        d = (
            dframe.loc[
                dframe["cell_number"] == cell
            ]
            .sort_values("mid_time")
        )

        height_m = float(
            d["height_relative_to_frame_bottom_m"].iloc[0]
        )
        
        plot_with_gaps(
            ax,
            d["mid_time"],
            d["ig_coherence_horizontal_weighted"],
            label=r"Horizontal speed $U_h$",
        )

        # plot_with_gaps(
        #     ax,
        #     d["mid_time"],
        #     d["ig_coherence_cross_weighted"],
        #     label=r"Cross-shore $U$",
        # )

        # plot_with_gaps(
        #     ax,
        #     d["mid_time"],
        #     d["ig_coherence_along_weighted"],
        #     label=r"Alongshore $V$",
        # )

        ax.set_ylim(0, 1)
        ax.set_ylabel(r"$\gamma^2$")
        ax.grid(True, alpha=0.25)

        ax.text(
            0.01, 0.90,
            f"Cell {int(cell)}, z={height_m:.3f} m",
            transform=ax.transAxes,
            va="top",
        )

        ax.legend(
            loc="upper right",
            ncol=2,
        )

    axes[-1].set_xlabel("Time")

    fig.suptitle(
        "IG-band ADV-pressure / ADCP-velocity coherence\n"
        f"{frame_id}, {ig_low_hz:.3f}-{ig_high_hz:.3f} Hz"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.show()


# ============================================================
# FIGURE 2
# CROSS / ALONG / PRINCIPAL-AXIS IG PHASE VS TIME
# ============================================================

for frame_id in frames:
    dframe = (
        diagnostics.loc[
            diagnostics[
                "frame_id"
            ]
            == frame_id
        ]
        .copy()
    )

    cells = sorted(
        dframe[
            "cell_number"
        ].unique()
    )

    fig, axes = plt.subplots(
        len(cells),
        1,
        figsize=(
            12,
            2.7 * len(cells),
        ),
        sharex=True,
    )

    if len(cells) == 1:
        axes = [axes]

    for ax, cell in zip(
        axes,
        cells,
    ):
        d = (
            dframe.loc[
                dframe[
                    "cell_number"
                ]
                == cell
            ]
            .sort_values(
                "mid_time"
            )
        )

        ax.scatter(
            d["mid_time"],
            d[
                "ig_phase_cross_deg"
            ],
            s=12,
            label=r"Cross-shore $U$",
        )

        ax.scatter(
            d["mid_time"],
            d[
                "ig_phase_along_deg"
            ],
            s=12,
            label=r"Alongshore $V$",
        )

        # ax.scatter(
        #     d["mid_time"],
        #     d[
        #         "ig_phase_axis_deg"
        #     ],
        #     s=12,
        #     label=r"Principal axis $U_a$",
        # )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.axhline(
            180,
            linestyle=":",
            linewidth=1,
        )

        ax.axhline(
            -180,
            linestyle=":",
            linewidth=1,
        )

        ax.set_ylim(
            -180,
            180,
        )

        ax.set_ylabel(
            "Phase (deg)"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.text(
            0.01,
            0.90,
            f"Cell {cell}",
            transform=
                ax.transAxes,
            va="top",
        )

        ax.legend(
            loc="upper right",
            ncol=3,
        )

    axes[-1].set_xlabel(
        "Time"
    )

    fig.suptitle(
        "IG-band ADV-pressure / ADCP-velocity phase\n"
        f"{frame_id}, {ig_low_hz:.3f}-{ig_high_hz:.3f} Hz"
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


# # ============================================================
# # FIGURE 3
# # IG PRINCIPAL-AXIS ORIENTATION + VARIANCE RATIO
# # ============================================================

# for frame_id in frames:
#     dframe = (
#         diagnostics.loc[
#             diagnostics[
#                 "frame_id"
#             ]
#             == frame_id
#         ]
#         .copy()
#     )

#     cells = sorted(
#         dframe[
#             "cell_number"
#         ].unique()
#     )

#     fig, axes = plt.subplots(
#         len(cells),
#         2,
#         figsize=(
#             12,
#             2.6 * len(cells),
#         ),
#         sharex="col",
#     )

#     if len(cells) == 1:
#         axes = np.array(
#             [axes]
#         )

#     for row, cell in enumerate(
#         cells
#     ):
#         d = (
#             dframe.loc[
#                 dframe[
#                     "cell_number"
#                 ]
#                 == cell
#             ]
#             .sort_values(
#                 "mid_time"
#             )
#         )

#         axes[row, 0].scatter(
#             d["mid_time"],
#             d[
#                 "ig_principal_axis_deg"
#             ],
#             s=12,
#         )

#         axes[row, 0].set_ylim(
#             0,
#             180,
#         )

#         axes[row, 0].set_ylabel(
#             rf"Cell {cell}" "\n"
#             r"$\theta_{\mathrm{IG}}$ (deg)"
#         )

#         axes[row, 0].grid(
#             True,
#             alpha=0.25,
#         )

#         axes[row, 1].plot(
#             d["mid_time"],
#             d[
#                 "ig_axis_variance_ratio"
#             ],
#         )

#         axes[row, 1].axhline(
#             1.0,
#             linestyle="--",
#             linewidth=1,
#         )

#         axes[row, 1].set_ylim(
#             bottom=0
#         )

#         axes[row, 1].set_ylabel(
#             r"$\lambda_1/\lambda_2$"
#         )

#         axes[row, 1].grid(
#             True,
#             alpha=0.25,
#         )

#     axes[-1, 0].set_xlabel(
#         "Time"
#     )

#     axes[-1, 1].set_xlabel(
#         "Time"
#     )

#     axes[0, 0].set_title(
#         "IG principal-axis orientation"
#     )

#     axes[0, 1].set_title(
#         "IG major/minor variance ratio"
#     )

#     fig.suptitle(
#         f"IG horizontal velocity-axis diagnostics ({frame_id})"
#     )

#     fig.tight_layout(
#         rect=[
#             0,
#             0,
#             1,
#             0.95,
#         ]
#     )

#     plt.show()

# ============================================================
# FOUR-PERIOD COMPARISON
# MEDIAN PRESSURE PSD + MEDIAN BURST COHERENCE WITH 25-75% BAND
# ONE FIGURE PER ADCP CELL
# ============================================================

periods = {
    "Early Apr IG event": (
        pd.Timestamp("2018-04-05", tz="UTC"),
        pd.Timestamp("2018-04-07", tz="UTC"),
    ),
    "Mid Apr IG event": (
        pd.Timestamp("2018-04-17", tz="UTC"),
        pd.Timestamp("2018-04-19", tz="UTC"),
    ),
    "May IG event": (
        pd.Timestamp("2018-05-10", tz="UTC"),
        pd.Timestamp("2018-05-12", tz="UTC"),
    ),
    "Calm": (
        pd.Timestamp("2018-05-06", tz="UTC"),
        pd.Timestamp("2018-05-8", tz="UTC"),
    ),
}

cs = cross_spectra.copy()
cs["mid_time"] = ensure_ns_utc(cs["mid_time"])
cs["frequency_bin_hz"] = cs["frequency_hz"].round(5)

frame_id = "F1"
cs = cs.loc[
    (cs["frame_id"] == frame_id)
    & (cs["frequency_bin_hz"] >= ig_low_hz)
    & (cs["frequency_bin_hz"] <= ig_high_hz)
].copy()


# ============================================================
# PLOT
# ============================================================

for cell in sorted(cs["cell_number"].unique()):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
    )

    for period_name, (start, end) in periods.items():

        d = cs.loc[
            (cs["cell_number"] == cell)
            & (cs["mid_time"] >= start)
            & (cs["mid_time"] < end)
        ].copy()

        if d.empty:
            print(f"No data: Cell {cell}, {period_name}")
            continue

        summary = (
            d.groupby("frequency_bin_hz")
            .agg(
                Spp_median=(
                    "pressure_head_psd_m2_hz",
                    "median",
                ),

                coh_pu_median=(
                    "pressure_cross_velocity_coherence",
                    "median",
                ),
                coh_pu_q25=(
                    "pressure_cross_velocity_coherence",
                    lambda x: x.quantile(0.25),
                ),
                coh_pu_q75=(
                    "pressure_cross_velocity_coherence",
                    lambda x: x.quantile(0.75),
                ),
                
                coh_pv_median=(
                    "pressure_along_velocity_coherence",
                    "median",
                ),
                coh_pv_q25=(
                    "pressure_along_velocity_coherence",
                    lambda x: x.quantile(0.25),
                ),
                coh_pv_q75=(
                    "pressure_along_velocity_coherence",
                    lambda x: x.quantile(0.75),
                ),

                n_bursts=(
                    "pressure_analysis_block_number",
                    "nunique",
                ),
            )
            .reset_index()
            .sort_values("frequency_bin_hz")
        )

        f = summary["frequency_bin_hz"].to_numpy()

        n_bursts = int(
            summary["n_bursts"].max()
        )

        # ----------------------------------------------------
        # Pressure PSD
        # ----------------------------------------------------

        line, = axes[0].semilogy(
            f,
            summary["Spp_median"],
            label=f"{period_name} (N={n_bursts})",
        )

        color = line.get_color()

        # ----------------------------------------------------
        # Cross-shore U coherence
        # solid = median
        # shaded = 25-75%
        # ----------------------------------------------------

        axes[1].plot(
            f,
            summary["coh_pu_median"],
            color=color,
            linestyle="-",
            label=period_name + r" $U$",
        )

        # axes[1].fill_between(
        #     f,
        #     summary["coh_pu_q25"],
        #     summary["coh_pu_q75"],
        #     color=color,
        #     alpha=0.15,
        # )
        
        axes[1].plot(
            f,
            summary["coh_pv_median"],
            color=color,
            linestyle="--",
            label=period_name + r" $V$",
        )

        # axes[1].fill_between(
        #     f,
        #     summary["coh_pv_q25"],
        #     summary["coh_pv_q75"],
        #     color=color,
        #     alpha=0.15,
        # )

    # ========================================================
    # FORMAT
    # ========================================================

    axes[0].set_ylabel(
        r"Median $S_{pp}$ (m$^2$/Hz)"
    )
    axes[0].grid(
        True,
        which="both",
        alpha=0.25,
    )
    axes[0].legend()

    axes[1].set_ylabel(
        r"Coherence $\gamma^2$"
    )
    axes[1].set_xlabel(
        "Frequency (Hz)"
    )
    axes[1].set_ylim(0, 1)
    axes[1].grid(
        True,
        alpha=0.25,
    )
    axes[1].legend(
        ncol=2,
    )

    for ax in axes:
        ax.set_xlim(
            ig_low_hz,
            ig_high_hz,
        )

    height_m = (
        cs.loc[
            cs["cell_number"] == cell,
            "height_relative_to_frame_bottom_m",
        ]
        .dropna()
        .iloc[0]
    )

    fig.suptitle(
        "IG pressure–velocity coherence within 4 periods\n"
        f"{frame_id}, Cell {int(cell)}, "
        f"z={height_m:.3f} m"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.show()