# -*- coding: utf-8 -*-
"""
Expected horizontal orbital-velocity spectra from pressure, compared with
measured ADCP-HR horizontal velocity spectra.

Linear finite-depth theory:
    Kp(f) = cosh(k z_p) / cosh(k h)
    S_eta(f) = S_phead(f) / Kp(f)^2

    Hu_eta(f,z) = omega * cosh(k z) / sinh(k h) -- transfer function
    S_U,expected(f,z) = |Hu_eta|^2 * S_eta(f)

Here z is height above bed. The measured horizontal velocity spectrum is
    S_U,measured = S_uu + S_vv
not the spectrum of sqrt(u^2 + v^2).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import newton
from scipy.signal import welch


# ============================================================
# USER SETTINGS
# ============================================================

input_folder = Path(r"C:\dev\Python\LongWaveAnalysis\Spectra")

velocity_file = input_folder / "all_cases_ADCP_HR_directional_velocity.parquet"
pressure_file = input_folder / "all_cases_ADCP_UP_directional_pressure.parquet"

# ADCP-HR cells to compare. Heights are read from the velocity Parquet.
cells_to_use = [1, 5, 9, 13]

gravity_m_s2 = 9.81

welch_segment_seconds = 512.0
overlap_fraction = 0.5

ig_low_hz, ig_high_hz = 0.005, 0.04
ss_low_hz, ss_high_hz = 0.04, 1.0

minimum_samples = 300
minimum_overlap_fraction = 0.95

plot_fmin_hz = 0.005
plot_fmax_hz = 1.0


# ============================================================
# BASIC HELPERS
# ============================================================

def extract_frame_id(values):
    return (
        values.astype(str)
        .str.extract(r"(F\d+)", expand=False)
        .str.upper()
    )


def ensure_ns_utc(series):
    return pd.to_datetime(
        series, utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")


def detect_fs(time):
    dt = (
        pd.Series(time)
        .diff()
        .dt.total_seconds()
        .to_numpy(float)
    )
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        return np.nan

    cutoff = np.nanpercentile(dt, 25)
    regular = dt[dt <= 1.5 * cutoff]
    if regular.size == 0:
        regular = dt

    return 1.0 / np.nanmedian(regular)


def welch_psd(values, fs):
    values = np.asarray(values, float)
    nperseg = min(
        int(round(welch_segment_seconds * fs)),
        len(values),
    )

    if nperseg < 16:
        return None, None

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
    )


def solve_wavenumber(frequency_hz, depth_m):
    """Solve omega^2 = g k tanh(kh) frequency by frequency."""
    f = np.asarray(frequency_hz, float)
    omega = 2 * np.pi * f
    k = np.zeros_like(f)

    positive = f > 0
    for i in np.where(positive)[0]:
        w = omega[i]
        k0 = max(w / np.sqrt(gravity_m_s2 * depth_m),
                 w**2 / gravity_m_s2)
        try:
            k[i] = newton(
                lambda kk: gravity_m_s2 * kk * np.tanh(kk * depth_m) - w**2,
                k0,
                maxiter=50,
            )
        except RuntimeError:
            # Robust Newton iteration fallback
            kk = k0
            for _ in range(50):
                kh = kk * depth_m
                fun = gravity_m_s2 * kk * np.tanh(kh) - w**2
                der = gravity_m_s2 * (
                    np.tanh(kh) + kh / np.cosh(kh)**2
                )
                kk_new = kk - fun / der
                if abs(kk_new - kk) < 1e-12:
                    break
                kk = kk_new
            k[i] = kk

    return k


def expected_velocity_psd_from_pressure(
    frequency_hz,
    pressure_head_psd,
    water_depth_m,
    pressure_height_m,
    velocity_height_m,
):
    """
    Convert pressure-head PSD directly to expected total horizontal
    orbital-velocity PSD at velocity_height_m using linear wave theory.
    """
    f = np.asarray(frequency_hz, float)
    spp = np.asarray(pressure_head_psd, float)

    k = solve_wavenumber(f, water_depth_m)
    omega = 2 * np.pi * f

    kp = np.ones_like(f)
    hu_eta = np.zeros_like(f)

    positive = f > 0

    kp[positive] = (
        np.cosh(k[positive] * pressure_height_m)
        / np.cosh(k[positive] * water_depth_m)
    )

    hu_eta[positive] = (
        omega[positive]
        * np.cosh(k[positive] * velocity_height_m)
        / np.sinh(k[positive] * water_depth_m)
    )

    seta = np.full_like(spp, np.nan)
    expected = np.full_like(spp, np.nan)

    valid = positive & np.isfinite(spp) & (kp > 0)
    seta[valid] = spp[valid] / kp[valid]**2
    expected[valid] = seta[valid] * hu_eta[valid]**2

    return expected, seta, kp, k


def band_rms(frequency_hz, psd, fmin, fmax):
    band = (
        (frequency_hz >= fmin)
        & (frequency_hz < fmax)
        & np.isfinite(psd)
    )
    if band.sum() < 2:
        return np.nan

    variance = np.trapezoid(
        psd[band],
        frequency_hz[band],
    )
    return np.sqrt(max(variance, 0.0))


def median_spectrum(df, value_column):
    """Median burst spectrum on rounded common frequency bins."""
    d = df[
        ["frequency_hz", value_column]
    ].dropna().copy()

    d["frequency_bin_hz"] = d["frequency_hz"].round(6)

    return (
        d.groupby("frequency_bin_hz")[value_column]
        .median()
        .reset_index()
        .rename(columns={"frequency_bin_hz": "frequency_hz"})
        .sort_values("frequency_hz")
    )


# ============================================================
# OPEN PARQUETS + DISCOVER CASES/BURSTS DIRECTLY
# ============================================================

velocity_dataset = ds.dataset(velocity_file, format="parquet")
pressure_dataset = ds.dataset(pressure_file, format="parquet")

required_velocity = {
    "case_id", "cell_number", "height_relative_to_frame_bottom_m",
    "time", "cross_velocity_detrended_m_s", "along_velocity_detrended_m_s",
}
required_pressure = {
    "case_id", "analysis_block_number", "time",
    "pressure_head_detrended_m", "mean_water_depth_m", "sensor_height_above_bed_m",
}

missing_v = required_velocity - set(velocity_dataset.schema.names)
missing_p = required_pressure - set(pressure_dataset.schema.names)
if missing_v:
    raise KeyError(f"Missing velocity columns: {sorted(missing_v)}")
if missing_p:
    raise KeyError(f"Missing pressure columns: {sorted(missing_p)}")

# Read only lightweight indexing columns from the Parquets.
v_index = velocity_dataset.to_table(
    columns=["case_id"]
).to_pandas().drop_duplicates()
v_index["frame_id"] = extract_frame_id(v_index["case_id"])

p_index = pressure_dataset.to_table(
    columns=["case_id", "analysis_block_number", "time"]
).to_pandas()
p_index["frame_id"] = extract_frame_id(p_index["case_id"])
p_index["time"] = ensure_ns_utc(p_index["time"])

velocity_case_lookup = (
    v_index.dropna()
    .groupby("frame_id")["case_id"]
    .apply(lambda x: sorted(x.astype(str).unique()))
    .to_dict()
)

# One row per pressure analysis block, with times derived directly
# from the pressure Parquet.
pressure_bursts = (
    p_index.dropna(subset=["frame_id", "case_id", "analysis_block_number", "time"])
    .groupby(["frame_id", "case_id", "analysis_block_number"], as_index=False)
    .agg(start_time=("time", "min"), end_time=("time", "max"))
)
pressure_bursts["mid_time"] = (
    pressure_bursts["start_time"]
    + (pressure_bursts["end_time"] - pressure_bursts["start_time"]) / 2
)

common_frames = sorted(
    set(velocity_case_lookup) & set(pressure_bursts["frame_id"].unique())
)
pressure_bursts = (
    pressure_bursts[
        pressure_bursts["frame_id"].isin(common_frames)
    ]
    .sort_values(["frame_id", "case_id", "start_time"])
    .reset_index(drop=True)
)


# ============================================================
# BURST ANALYSIS
# ============================================================

spectral_records = []
rms_records = []

pressure_columns = [
    "case_id",
    "analysis_block_number",
    "time",
    "pressure_head_detrended_m",
    "mean_water_depth_m",
    "sensor_height_above_bed_m",
]

velocity_columns = [
    "case_id",
    "cell_number",
    "height_relative_to_frame_bottom_m",
    "time",
    "cross_velocity_detrended_m_s",
    "along_velocity_detrended_m_s",
]

for _, meta in pressure_bursts.iterrows():

    frame_id = meta["frame_id"]
    pressure_case_id = str(meta["case_id"])
    block_number = int(meta["analysis_block_number"])

    velocity_cases = velocity_case_lookup.get(frame_id, [])
    if len(velocity_cases) != 1:
        raise RuntimeError(
            f"Expected one velocity case for {frame_id}, "
            f"found {velocity_cases}"
        )
    velocity_case_id = velocity_cases[0]

    # --------------------------------------------------------
    # Pressure burst
    # --------------------------------------------------------
    p_filter = (
        (ds.field("case_id") == pressure_case_id)
        & (ds.field("analysis_block_number") == block_number)
    )

    p = (
        pressure_dataset
        .to_table(columns=pressure_columns, filter=p_filter)
        .to_pandas()
    )
    
    z_pressure = float(
    np.nanmedian(p["sensor_height_above_bed_m"].to_numpy(dtype=float)))

    if p.empty:
        continue

    p["time"] = ensure_ns_utc(p["time"])
    p = (
        p.dropna(subset=["time", "pressure_head_detrended_m"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(p) < minimum_samples:
        continue

    fs_p = detect_fs(p["time"])
    if not np.isfinite(fs_p):
        continue

    f_p, spp = welch_psd(
        p["pressure_head_detrended_m"].to_numpy(float),
        fs_p,
    )
    if f_p is None:
        continue

    h = float(np.nanmedian(p["mean_water_depth_m"]))
    if not np.isfinite(h) or h <= 0:
        continue

    p_start = p["time"].iloc[0]
    p_end = p["time"].iloc[-1]
    p_mid = p_start + (p_end - p_start) / 2

    # --------------------------------------------------------
    # Simultaneous selected ADCP-HR cells
    # --------------------------------------------------------
    v_filter = (
        (ds.field("case_id") == velocity_case_id)
        & (ds.field("time") >= p_start.to_pydatetime())
        & (ds.field("time") <= p_end.to_pydatetime())
        & ds.field("cell_number").isin(cells_to_use)
    )

    v = (
        velocity_dataset
        .to_table(columns=velocity_columns, filter=v_filter)
        .to_pandas()
    )

    if v.empty:
        continue

    v["time"] = ensure_ns_utc(v["time"])

    for cell in cells_to_use:

        vc = (
            v.loc[v["cell_number"] == cell]
            .dropna(subset=[
                "time",
                "cross_velocity_detrended_m_s",
                "along_velocity_detrended_m_s",
            ])
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        if len(vc) < minimum_samples:
            continue

        # Require velocity to span nearly the whole pressure burst.
        overlap = (
            min(vc["time"].iloc[-1], p_end)
            - max(vc["time"].iloc[0], p_start)
        ).total_seconds()

        p_duration = (p_end - p_start).total_seconds()
        if p_duration <= 0 or overlap / p_duration < minimum_overlap_fraction:
            continue

        z = float(
            np.nanmedian(
                vc["height_relative_to_frame_bottom_m"]
            )
        )

        if not (0 <= z < h):
            continue

        # ----------------------------------------------------
        # Measured total horizontal velocity PSD = Suu + Svv
        # ----------------------------------------------------
        fs_v = detect_fs(vc["time"])
        if not np.isfinite(fs_v):
            continue

        f_v, suu = welch_psd(
            vc["cross_velocity_detrended_m_s"].to_numpy(float),
            fs_v,
        )
        _, svv = welch_psd(
            vc["along_velocity_detrended_m_s"].to_numpy(float),
            fs_v,
        )

        if f_v is None:
            continue

        su_measured = suu + svv

        # ----------------------------------------------------
        # Expected orbital-velocity PSD from pressure
        # ----------------------------------------------------
        su_expected, seta, kp, k = (
            expected_velocity_psd_from_pressure(
                f_p,
                spp,
                water_depth_m=h,
                pressure_height_m=z_pressure,
                velocity_height_m=z,
            )
        )

        # Keep only direct pressure-resolved range.
        expected_valid = (
            (f_p >= plot_fmin_hz)
            & (f_p <= plot_fmax_hz)
        )

        spectral_records.append(
            pd.DataFrame({
                "frame_id": frame_id,
                "pressure_case_id": pressure_case_id,
                "velocity_case_id": velocity_case_id,
                "analysis_block_number": block_number,
                "cell_number": cell,
                "height_m": z,
                "mid_time": p_mid,
                "water_depth_m": h,
                "frequency_hz": f_p[expected_valid],
                "pressure_head_psd_m2_hz": spp[expected_valid],
                "surface_elevation_psd_m2_hz": seta[expected_valid],
                "pressure_response_kp": kp[expected_valid],
                "expected_horizontal_velocity_psd_m2_s2_hz":
                    su_expected[expected_valid],
            })
        )

        # Store measured spectrum separately on its native frequency grid.
        measured_valid = (
            (f_v >= plot_fmin_hz)
            & (f_v <= plot_fmax_hz)
        )

        measured_frame = pd.DataFrame({
            "frame_id": frame_id,
            "cell_number": cell,
            "height_m": z,
            "mid_time": p_mid,
            "frequency_hz": f_v[measured_valid],
            "measured_horizontal_velocity_psd_m2_s2_hz":
                su_measured[measured_valid],
        })

        # Append as another spectral record; columns align on concat.
        spectral_records.append(measured_frame)

        # ----------------------------------------------------
        # Band RMS: same frequency limits for both sources
        # ----------------------------------------------------
        rms_records.append({
            "frame_id": frame_id,
            "pressure_case_id": pressure_case_id,
            "velocity_case_id": velocity_case_id,
            "analysis_block_number": block_number,
            "cell_number": cell,
            "height_m": z,
            "mid_time": p_mid,
            "water_depth_m": h,

            "expected_ig_urms_m_s":
                band_rms(f_p, su_expected, ig_low_hz, ig_high_hz),
            "measured_ig_urms_m_s":
                band_rms(f_v, su_measured, ig_low_hz, ig_high_hz),

            "expected_ss_urms_m_s":
                band_rms(f_p, su_expected, ss_low_hz, ss_high_hz),
            "measured_ss_urms_m_s":
                band_rms(f_v, su_measured, ss_low_hz, ss_high_hz),
        })


# ============================================================
# BUILD RESULTS
# ============================================================

spectra = pd.concat(
    spectral_records,
    ignore_index=True,
) if spectral_records else pd.DataFrame()

rms = pd.DataFrame(rms_records)

if spectra.empty or rms.empty:
    raise RuntimeError(
        "No matched pressure/velocity bursts passed the criteria."
    )

print("\nMatched burst counts:")
print(
    rms.groupby(["frame_id", "cell_number"])
    .size()
)

print("\nCell heights [m above bed]:")
print(
    rms.groupby(["frame_id", "cell_number"])["height_m"]
    .median()
)


# ============================================================
# FIGURE 1: MEDIAN EXPECTED VS MEASURED SPECTRA
# ============================================================

for frame_id in sorted(rms["frame_id"].unique()):

    cells = sorted(
        rms.loc[rms["frame_id"] == frame_id, "cell_number"]
        .unique()
    )

    fig, axes = plt.subplots(
        len(cells), 1,
        figsize=(10, 2.8 * len(cells)),
        sharex=True,
        sharey=True,
    )

    if len(cells) == 1:
        axes = [axes]

    for ax, cell in zip(axes, cells):

        d = spectra.loc[
            (spectra["frame_id"] == frame_id)
            & (spectra["cell_number"] == cell)
        ]

        expected = median_spectrum(
            d,
            "expected_horizontal_velocity_psd_m2_s2_hz",
        )
        measured = median_spectrum(
            d,
            "measured_horizontal_velocity_psd_m2_s2_hz",
        )

        z = float(
            rms.loc[
                (rms["frame_id"] == frame_id)
                & (rms["cell_number"] == cell),
                "height_m",
            ].median()
        )

        ax.loglog(
            measured["frequency_hz"],
            measured["measured_horizontal_velocity_psd_m2_s2_hz"],
            label="Measured $S_{uu}+S_{vv}$",
        )

        ax.loglog(
            expected["frequency_hz"],
            expected["expected_horizontal_velocity_psd_m2_s2_hz"],
            "--",
            linewidth=2,
            label="Expected from pressure",
        )

        ax.axvline(ig_high_hz, color="0.5", linestyle=":")
        ax.set_ylabel(
            r"$S_U$ (m$^2$ s$^{-2}$/Hz)"
        )
        ax.grid(True, which="both", alpha=0.25)

        ax.text(
            0.02, 0.08,
            f"Cell {int(cell)}, z={z:.3f} m",
            transform=ax.transAxes,
        )

    axes[0].legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(plot_fmin_hz, plot_fmax_hz)

    fig.suptitle(
        f"{frame_id}: median horizontal velocity spectrum\n"
        "measured vs linear-wave prediction from pressure"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
# FIGURE 2: TIME-VARYING IG + SS URMS
# ============================================================

for frame_id in sorted(rms["frame_id"].unique()):

    cells = sorted(
        rms.loc[rms["frame_id"] == frame_id, "cell_number"]
        .unique()
    )

    fig, axes = plt.subplots(
        len(cells), 2,
        figsize=(14, 2.7 * len(cells)),
        sharex=True,
        sharey="col",
    )

    if len(cells) == 1:
        axes = np.array([axes])

    for row, cell in enumerate(cells):

        d = (
            rms.loc[
                (rms["frame_id"] == frame_id)
                & (rms["cell_number"] == cell)
            ]
            .sort_values("mid_time")
        )

        z = float(d["height_m"].median())

        for col, band in enumerate(["ig", "ss"]):

            ax = axes[row, col]

            ax.plot(
                d["mid_time"],
                d[f"measured_{band}_urms_m_s"],
                label="Measured",
            )

            ax.plot(
                d["mid_time"],
                d[f"expected_{band}_urms_m_s"],
                "--",
                linewidth=1.8,
                label="Expected from pressure",
            )

            ax.grid(True, alpha=0.25)
            ax.set_ylabel(
                rf"Cell {int(cell)}, z={z:.3f} m"
                "\n"
                r"$U_{\mathrm{RMS}}$ (m/s)"
            )

    axes[0, 0].set_title(
        f"IG: {ig_low_hz:.3f}-{ig_high_hz:.2f} Hz"
    )
    axes[0, 1].set_title(
        f"SS: {ss_low_hz:.2f}-{ss_high_hz:.2f} Hz"
    )

    axes[0, 0].legend()
    axes[0, 1].legend()

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    fig.suptitle(
        f"{frame_id}: measured vs pressure-predicted "
        "horizontal orbital-velocity RMS"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
