# -*- coding: utf-8 -*-
"""
Compute ADV01 cross-shore and alongshore velocity autospectra
using the instrument's natural continuous recording bursts.

Typical KG2 pattern:
    about 29.83 minutes of data
    about 10 seconds gap
    next recording burst

Long, approximately 59.83-minute segments are divided into
two approximately 29.83-minute analysis blocks.
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
    r"\ADV_F3_ADV05_rotated.parquet"
)

Case = "DVN F3 ADV05"

output_folder = Path(
    r"C:\dev\Python\LongWaveAnalysis\Spectra"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True,
)

fs = 16.0 # 16 Hz ADV

# A new continuous segment begins when dt is larger than this.
# Normal dt = 0.0625 s, while the observed burst gap is 10.0625 s.
gap_threshold_seconds = 1.0

# Welch settings.
segment_seconds = 512.0
overlap_fraction = 0.5

# Frequency bands.
ig_low_hz = 0.005
ig_high_hz = 0.05

ss_low_hz = 0.05
ss_high_hz = 1.0

# Require almost all velocity values in an analysis block to be finite.
minimum_valid_fraction = 0.98

plot_first_valid_block = True


# ============================================================
# FUNCTIONS
# ============================================================

def integrate_spectral_band(
    frequency,
    psd,
    lower_hz,
    upper_hz,
):
    """
    Integrate a PSD over one frequency band.

    Returns variance in (m/s)^2.
    """
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

    return np.trapezoid(
        psd[selected],
        frequency[selected],
    )


def calculate_autospectra(
    cross_shore,
    alongshore,
    fs,
    segment_seconds=512.0,
    overlap_fraction=0.5,
):
    """
    Calculate Welch autospectra for signed cross-shore and
    alongshore velocities.

    PSD units:
        (m/s)^2 / Hz
        
    velocity variance <- Amplitude u=Acos(2pi*ft); 
    Stormy state -> higher variance -> higher PSD
    """
    cross = np.asarray(cross_shore, dtype=np.float64)
    along = np.asarray(alongshore, dtype=np.float64)

    if cross.shape != along.shape:
        raise ValueError(
            "Cross-shore and alongshore arrays have different shapes."
        )

    if cross.ndim != 1:
        raise ValueError("Velocity arrays must be one-dimensional.")

    if not np.isfinite(cross).all():
        raise ValueError("Cross-shore velocity contains missing values.")

    if not np.isfinite(along).all():
        raise ValueError("Alongshore velocity contains missing values.")

    nperseg = int(round(segment_seconds * fs))
    nperseg = min(nperseg, len(cross))

    if nperseg < 16:
        raise ValueError("Analysis block is too short.")

    noverlap = int(round(overlap_fraction * nperseg))

    # noverlap must be smaller than nperseg.
    noverlap = min(noverlap, nperseg - 1)

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

    if not np.allclose(frequency, frequency_along):
        raise RuntimeError(
            "Cross-shore and alongshore frequency grids differ."
        )

    return frequency, cross_psd, along_psd


def identify_continuous_segments(
    df,
    gap_threshold_seconds=1.0,
):
    """
    Add a segment ID that changes whenever the timestamp gap
    exceeds gap_threshold_seconds.
    """
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
    """
    Divide one continuous segment into analysis blocks.

    Rules
    -----
    - A segment near 28,640 samples becomes one block.
    - A segment near 57,440 samples becomes two blocks.
    - Any final remainder is retained only when it is at least
      minimum_block_samples long.
    """
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

def polynomial_detrend(values, order=2):# remove the background mean flow
    """
    Remove a polynomial trend from one continuous velocity burst.

    order=0: remove mean
    order=1: remove linear trend
    order=2: remove quadratic trend
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite values.")

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

# ============================================================
# READ PROCESSED DATA
# ============================================================

print("Reading:")
print(input_file)

df = pd.read_parquet(input_file)

required_columns = {
    "time",
    "cross_shore",
    "alongshore",
}

missing = required_columns - set(df.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

df = df[
    [
        "time",
        "cross_shore",
        "alongshore",
    ]
].copy()

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
)

df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)

print("Start:", df["time"].iloc[0])
print("End:  ", df["time"].iloc[-1])
print("Rows:", len(df))

print("\nMissing values:")
print(
    df[
        [
            "cross_shore",
            "alongshore",
        ]
    ].isna().sum()
)


# ============================================================
# IDENTIFY NATURAL CONTINUOUS BURSTS
# ============================================================

df = identify_continuous_segments(
    df=df,
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

print("\nNumber of continuous segments:")
print(len(segment_summary))

print("\nSegment duration summary:")
print(
    segment_summary["duration_minutes"]
    .describe()
)

# Determine the nominal burst length automatically
nominal_burst_samples = int(
    segment_summary["sample_count"].mode().iloc[0]
)

print(
    "\nDetected nominal burst length:",
    nominal_burst_samples,
    "samples",
)

print(
    "Duration:",
    nominal_burst_samples / fs / 60,
    "minutes",
)

# Accept bursts that contain at least 95% of a full burst
minimum_block_fraction = 0.95

minimum_block_samples = int(
    round(
        nominal_burst_samples
        * minimum_block_fraction
    )
)

print(
    "Minimum accepted block:",
    minimum_block_samples,
    "samples",
)


# ============================================================
# GENERATE ANALYSIS BLOCKS
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

print("\nAnalysis blocks produced:")
print(len(analysis_blocks))


# ============================================================
# COMPUTE SPECTRA
# ============================================================

statistics_records = []
spectral_records = []

first_valid_spectrum = None
accepted_block_number = 0

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

    valid_mask = (
        block[
            [
                "cross_shore",
                "alongshore",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    valid_fraction = valid_mask.mean()

    accepted = (
        valid_fraction >= minimum_valid_fraction
        and sample_count >= minimum_block_samples
    )

    rejection_reason = ""

    if sample_count < minimum_block_samples:
        rejection_reason = "analysis block too short"

    elif valid_fraction < minimum_valid_fraction:
        rejection_reason = "too many missing velocity samples"

    if not accepted:
        statistics_records.append(
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
                "sample_count":
                    sample_count,
                "duration_minutes":
                    duration_minutes,
                "valid_fraction":
                    valid_fraction,
                "accepted":
                    False,
                "rejection_reason":
                    rejection_reason,
                "cross_ig_variance":
                    np.nan,
                "along_ig_variance":
                    np.nan,
                "cross_ss_variance":
                    np.nan,
                "along_ss_variance":
                    np.nan,
                "cross_ig_rms":
                    np.nan,
                "along_ig_rms":
                    np.nan,
                "cross_ss_rms":
                    np.nan,
                "along_ss_rms":
                    np.nan,
                "cross_along_ig_ratio":
                    np.nan,
                "cross_ig_ss_ratio":
                    np.nan,
            }
        )

        continue

    # Interpolate isolated missing samples within an accepted burst.
    cross_raw = (
        block["cross_shore"]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
        .to_numpy(dtype=np.float64)
    )
    
    along_raw = (
        block["alongshore"]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
        .to_numpy(dtype=np.float64)
    )

    # Retain the average current before detrending.
    mean_cross_current = np.mean(cross_raw)
    mean_along_current = np.mean(along_raw)
    
    # Remove a quadratic background from the complete burst.
    cross, cross_background = polynomial_detrend(
        cross_raw,
        order=2,
    )
    
    along, along_background = polynomial_detrend(
        along_raw,
        order=2,
    )
    
    # how much the fitted background current changes through each burst
    cross_background_change = (
        cross_background[-1]
        - cross_background[0]
    )
    
    along_background_change = (
        along_background[-1]
        - along_background[0]
    )

    # Calculate PSD from already detrended signals.
    frequency, cross_psd, along_psd = calculate_autospectra(
        cross_shore=cross,
        alongshore=along,
        fs=fs,
        segment_seconds=segment_seconds,
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
        ss_high_hz,
    )

    along_ss_variance = integrate_spectral_band(
        frequency,
        along_psd,
        ss_low_hz,
        ss_high_hz,
    )

    cross_ig_rms = np.sqrt(cross_ig_variance)
    along_ig_rms = np.sqrt(along_ig_variance)

    cross_ss_rms = np.sqrt(cross_ss_variance)
    along_ss_rms = np.sqrt(along_ss_variance)

    if (
        np.isfinite(along_ig_variance)
        and along_ig_variance > 0
    ):
        cross_along_ig_ratio = (
            cross_ig_variance
            / along_ig_variance
        )
    else:
        cross_along_ig_ratio = np.nan

    if (
        np.isfinite(cross_ss_variance)
        and cross_ss_variance > 0
    ):
        cross_ig_ss_ratio = (
            cross_ig_variance
            / cross_ss_variance
        )
    else:
        cross_ig_ss_ratio = np.nan

    statistics_records.append(
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
            "sample_count":
                sample_count,
            "duration_minutes":
                duration_minutes,
            "valid_fraction":
                valid_fraction,
            "accepted":
                True,
            "rejection_reason":
                "",
            "cross_ig_variance":
                cross_ig_variance,
            "along_ig_variance":
                along_ig_variance,
            "cross_ss_variance":
                cross_ss_variance,
            "along_ss_variance":
                along_ss_variance,
            "cross_ig_rms":
                cross_ig_rms,
            "along_ig_rms":
                along_ig_rms,
            "cross_ss_rms":
                cross_ss_rms,
            "along_ss_rms":
                along_ss_rms,
            "cross_along_ig_ratio":
                cross_along_ig_ratio,
            "cross_ig_ss_ratio":
                cross_ig_ss_ratio,
            "mean_cross_current": 
                mean_cross_current,
            "mean_along_current": 
                mean_along_current,
        }
    )

    block_spectrum = pd.DataFrame(
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
            "frequency_hz":
                frequency,
            "cross_psd":
                cross_psd,
            "along_psd":
                along_psd,
        }
    )

    spectral_records.append(block_spectrum)

    if first_valid_spectrum is None:
        first_valid_spectrum = block_spectrum.copy()

    accepted_block_number += 1


# ============================================================
# SAVE RESULTS
# ============================================================

statistics = pd.DataFrame(
    statistics_records
)

if spectral_records:
    spectra = pd.concat(
        spectral_records,
        ignore_index=True,
    )
else:
    spectra = pd.DataFrame()

statistics_file = (
    output_folder
    / f"{Case}""_burst_spectral_statistics.csv"
)

spectra_file = (
    output_folder
    / f"{Case}""_burst_velocity_spectra.pkl"
)

segment_file = (
    output_folder
    / f"{Case}""_segment_summary.csv"
)

statistics.to_csv(
    statistics_file,
    index=False,
)

spectra.to_pickle(
    spectra_file,
)

segment_summary.to_csv(
    segment_file,
    index=False,
)

print("\nSaved segment summary:")
print(segment_file)

print("\nSaved spectral statistics:")
print(statistics_file)

print("\nSaved spectra:")
print(spectra_file)

print("\nAccepted blocks:")
print(statistics["accepted"].sum())

print("Rejected blocks:")
print((~statistics["accepted"]).sum())


# # ============================================================
# # PLOT FIRST ACCEPTED BURST
# # ============================================================
# if (
#     plot_first_valid_block
#     and first_valid_spectrum is not None
# ):
#     positive = (
#         first_valid_spectrum["frequency_hz"] > 0
#     )

#     plot_data = first_valid_spectrum.loc[
#         positive
#     ]

#     plt.figure(figsize=(9, 6))

#     plt.loglog(
#         plot_data["frequency_hz"],
#         plot_data["cross_psd"],
#         label="Cross-shore",
#     )

#     plt.loglog(
#         plot_data["frequency_hz"],
#         plot_data["along_psd"],
#         label="Alongshore",
#     )

#     plt.axvspan(
#         ig_low_hz,
#         ig_high_hz,
#         alpha=0.20,
#         label="IG band",
#     )

#     plt.axvspan(
#         ss_low_hz,
#         ss_high_hz,
#         alpha=0.10,
#         label="Sea-swell band",
#     )

#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel(r"Velocity PSD ((m/s)$^2$/Hz)")

#     plt.title(
#         "ADV01 horizontal velocity autospectra\n"
#         f"{first_valid_spectrum['start_time'].iloc[0]}"
#     )

#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
# ========================================
# Evaluate statistics
# ========================================
accepted = statistics.loc[statistics["accepted"]].copy()

accepted["cross_ig_ss_ratio"] = (
    accepted["cross_ig_variance"]
    / accepted["cross_ss_variance"]
)

accepted["cross_fraction_ig"] = (
    accepted["cross_ig_variance"]
    / (
        accepted["cross_ig_variance"]
        + accepted["along_ig_variance"]
    )
)

print(
    accepted[
        [
            "cross_ig_rms",
            "along_ig_rms",
            "cross_ss_rms",
            "cross_along_ig_ratio",
            "cross_ig_ss_ratio",
            "cross_fraction_ig",
        ]
    ].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
)
        
# Plot temporal evolution of RMS u_ig and v_ig
accepted["mid_time"] = (
    accepted["start_time"]
    + (
        accepted["end_time"]
        - accepted["start_time"]
    ) / 2
)
accepted = accepted.sort_values("mid_time")

dt = accepted["mid_time"].diff().dt.total_seconds()

accepted["cross_ig_rms_plot"] = accepted["cross_ig_rms"]
accepted["along_ig_rms_plot"] = accepted["along_ig_rms"]

large_gap = dt > 3600

accepted.loc[
    large_gap,
    [
        "cross_ig_rms_plot",
        "along_ig_rms_plot",
    ],
] = np.nan

plt.figure(figsize=(12, 5))
plt.plot(
    accepted["mid_time"],
    accepted["cross_ig_rms_plot"],
    label="Cross-shore IG RMS",
)
plt.plot(
    accepted["mid_time"],
    accepted["along_ig_rms_plot"],
    label="Alongshore IG RMS",
)
plt.xlabel("Time")
plt.ylabel("IG velocity RMS (m/s)")
plt.title(f"{Case}")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot the most energetic IG burst
# ============================================================
accepted_statistics = statistics.loc[
    statistics["accepted"]
].copy()

# Total horizontal IG variance for each burst.
accepted_statistics["total_horizontal_ig_variance"] = (
    accepted_statistics["cross_ig_variance"]
    + accepted_statistics["along_ig_variance"]
)

# Find the most energetic accepted burst.
most_energetic_row = accepted_statistics.loc[
    accepted_statistics["total_horizontal_ig_variance"].idxmax()
]

# second most
# accepted_statistics = accepted_statistics.sort_values(
#     "total_horizontal_ig_variance",
#     ascending=False,
# )

# second_energetic_row = accepted_statistics.iloc[1]

most_energetic_block_number = int(
    most_energetic_row["analysis_block_number"]
)

most_energetic_spectrum = spectra.loc[
    spectra["analysis_block_number"]
    == most_energetic_block_number
].copy()

if most_energetic_spectrum.empty:
    raise RuntimeError(
        "The selected energetic burst was not found in spectra."
    )

positive = (
    most_energetic_spectrum["frequency_hz"] > 0
)

plot_data = most_energetic_spectrum.loc[
    positive
]

plt.figure(figsize=(9, 6))

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["cross_psd"],
    label="Cross-shore",
)

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["along_psd"],
    label="Alongshore",
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

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Velocity PSD ((m/s)$^2$/Hz)")

plt.title(
    "Most energetic horizontal IG burst " 
    f"({Case})"
    "\n"
    f"{most_energetic_row['start_time']} to "
    f"{most_energetic_row['end_time']}"
)

plt.legend()
plt.tight_layout()
plt.show()

# print("\nMost energetic IG burst:")
# print("Block number:", most_energetic_block_number)
# print("Start:", most_energetic_row["start_time"])
# print("End:", most_energetic_row["end_time"])
# print(
#     "Cross-shore IG RMS:",
#     most_energetic_row["cross_ig_rms"],
#     "m/s",
# )
# print(
#     "Alongshore IG RMS:",
#     most_energetic_row["along_ig_rms"],
#     "m/s",
# )
# print(
#     "Total horizontal IG RMS:",
#     np.sqrt(
#         most_energetic_row[
#             "total_horizontal_ig_variance"
#         ]
#     ),
#     "m/s",
# )

# Plot median spectrum
median_spectrum = (
    spectra
    .groupby("frequency_hz", as_index=False)
    .agg(
        cross_psd_median=("cross_psd", "median"),
        along_psd_median=("along_psd", "median"),
        cross_psd_q25=("cross_psd", lambda x: x.quantile(0.25)),
        cross_psd_q75=("cross_psd", lambda x: x.quantile(0.75)),
        along_psd_q25=("along_psd", lambda x: x.quantile(0.25)),
        along_psd_q75=("along_psd", lambda x: x.quantile(0.75)),
    )
)

positive = median_spectrum["frequency_hz"] > 0
plot_data = median_spectrum.loc[positive]

plt.figure(figsize=(9, 6))

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["cross_psd_median"],
    label="Cross-shore median",
)

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["along_psd_median"],
    label="Alongshore median",
)

plt.fill_between(
    plot_data["frequency_hz"],
    plot_data["cross_psd_q25"],
    plot_data["cross_psd_q75"],
    color="grey",
    alpha=0.2,
)

plt.axvspan(0.005, 0.05, alpha=0.20, label="IG band")
plt.axvspan(0.05, 1.00, alpha=0.10, label="Sea swell band")

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Velocity PSD ((m/s)$^2$/Hz)")
plt.title("Median velocity autospectra " f"({Case})")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot mean spectrum
# ============================================================

mean_spectrum = (
    spectra
    .groupby("frequency_hz", as_index=False)
    .agg(
        cross_psd_mean=("cross_psd", "mean"),
        along_psd_mean=("along_psd", "mean"),
        cross_psd_std=("cross_psd", "std"),
        along_psd_std=("along_psd", "std"),
    )
)

positive = mean_spectrum["frequency_hz"] > 0
plot_data = mean_spectrum.loc[positive]

plt.figure(figsize=(9, 6))

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["cross_psd_mean"],
    label="Cross-shore mean",
)

plt.loglog(
    plot_data["frequency_hz"],
    plot_data["along_psd_mean"],
    label="Alongshore mean",
)

plt.axvspan(
    0.005,
    0.05,
    alpha=0.20,
    label="IG band",
)

plt.axvspan(
    0.05,
    1.00,
    alpha=0.10,
    label="Sea swell band",
)

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Velocity PSD ((m/s)$^2$/Hz)")
plt.title("Mean velocity autospectra " f"({Case})")

plt.legend()
plt.tight_layout()
plt.show()

