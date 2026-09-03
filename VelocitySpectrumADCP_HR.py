# -*- coding: utf-8 -*-
"""ADCP-HR velocity spectra and IG/SS RMS for selected cells.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import welch, csd

# ------------------------- USER SETTINGS -------------------------
input_file = Path(r"C:\dev\Python\LongWaveAnalysis\Processed\DVN_F1_ADCP_HR_rotated.parquet")
case_id = "DVN_F1_ADCP-HR"
case_label = "DVN F1 ADCP-HR"
output_folder = Path(r"C:\dev\Python\LongWaveAnalysis\Spectra")
output_folder.mkdir(parents=True, exist_ok=True)

selected_cell_numbers = [1, 5, 9, 13]
gap_threshold_seconds = 2.0
maximum_interpolation_gap_seconds = 2.0
minimum_valid_fraction = 0.98
minimum_block_fraction = 0.95
minimum_velocity_std_m_s = 1e-4
welch_segment_seconds = 512.0
overlap_fraction = 0.5
detrend_order = 2
ig_low_hz, ig_high_hz = 0.005, 0.05
ss_low_hz, ss_high_hz = 0.05, 1.0
plot_gap_seconds = 3600.0

# --------------------------- FUNCTIONS ---------------------------
def identify_velocity_columns(columns):
    pat = re.compile(
        r"^(cross_shore|alongshore)_cell(?P<cell>\d+)_id(?P<id>\d+)_"
        r"z(?P<sign>[pm])(?P<mm>\d+)mm_m_s$"
    )
    found = {}
    for col in columns:
        m = pat.match(col)
        if not m:
            continue
        mm = int(m.group("mm")) * (-1 if m.group("sign") == "m" else 1)
        key = (int(m.group("cell")), int(m.group("id")), mm)
        d = found.setdefault(key, {
            "cell_number": key[0], "cell_id": key[1], "z_bin_m": mm / 1000.0
        })
        d[m.group(1)] = col
    cells = [d for d in found.values() if {"cross_shore", "alongshore"} <= d.keys()]
    if not cells:
        raise KeyError("No paired cross-shore/alongshore ADCP-HR columns found.")
    return sorted(cells, key=lambda d: d["cell_number"])


def detect_fs(time):
    dt = pd.Series(time).diff().dt.total_seconds().to_numpy(float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        raise ValueError("Could not determine sampling interval.")
    q25 = np.nanpercentile(dt, 25)
    regular = dt[dt <= 1.5 * q25]
    median_dt = float(np.nanmedian(regular if len(regular) else dt))
    return 1.0 / median_dt, median_dt


def add_segment_id(df):
    dt = df["time"].diff().dt.total_seconds()
    new = dt.isna() | (dt > gap_threshold_seconds) | (dt <= 0)
    out = df.copy()
    out["segment_id"] = new.cumsum()
    return out


def split_blocks(segment, nominal_n, minimum_n):
    blocks, start, sub = [], 0, 0
    while len(segment) - start >= nominal_n:
        b = segment.iloc[start:start + nominal_n].copy()
        b["subblock_number"] = sub
        blocks.append(b)
        start += nominal_n
        sub += 1
    if len(segment) - start >= minimum_n:
        b = segment.iloc[start:].copy()
        b["subblock_number"] = sub
        blocks.append(b)
    return blocks


def interp_short(series, limit):
    return pd.to_numeric(series, errors="coerce").interpolate(
        method="linear", limit=limit, limit_area="inside"
    )


def poly_detrend(values, order=2):
    y = np.asarray(values, float)
    x = np.linspace(-1, 1, len(y))
    return y - np.polyval(np.polyfit(x, y, order), x)


def spectra_uv(u, v, fs):
    nperseg = min(int(round(welch_segment_seconds * fs)), len(u))
    if nperseg < 16:
        raise ValueError("Analysis block too short for Welch analysis.")
    noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)
    kw = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
              detrend=False, scaling="density", return_onesided=True)
    f, Suu = welch(u, **kw)
    f2, Svv = welch(v, **kw)
    f3, Suv = csd(u, v, **kw)
    if not (np.allclose(f, f2) and np.allclose(f, f3)):
        raise RuntimeError("Spectral frequency grids differ.")
    return f, Suu, Svv, Suv


def band_integral(f, S, lo, hi):
    m = np.isfinite(f) & np.isfinite(S) & (f >= lo) & (f < hi)
    return np.trapezoid(S[m], f[m]) if m.sum() >= 2 else np.nan


def gap_plot_series(d, column):
    y = d[column].copy()
    y.loc[d["mid_time"].diff().dt.total_seconds() > plot_gap_seconds] = np.nan
    return y

# ----------------------- DISCOVER / LOAD DATA -----------------------
parquet = pq.ParquetFile(input_file)
cols = parquet.schema_arrow.names
if "time" not in cols:
    raise KeyError("Parquet file does not contain 'time'.")

all_cells = identify_velocity_columns(cols)
for c in all_cells:
    c["height_relative_to_frame_bottom_m"] = c["z_bin_m"]

available = {c["cell_number"] for c in all_cells}
missing = [n for n in selected_cell_numbers if n not in available]
if missing:
    raise ValueError(f"Selected cells not found: {missing}; available={sorted(available)}")

cells = sorted(
    [c for c in all_cells if c["cell_number"] in selected_cell_numbers],
    key=lambda c: selected_cell_numbers.index(c["cell_number"]),
)

print("Cells used for spectral analysis:")
for c in cells:
    print(f"  Cell {c['cell_number']}: z={c['z_bin_m']:.3f} m")

required = ["time"] + [x for c in cells for x in (c["cross_shore"], c["alongshore"])]
df = pd.read_parquet(input_file, columns=required)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
for col in required[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

fs, dt = detect_fs(df["time"])
print(f"Sampling: dt={dt:g} s, fs={fs:g} Hz")
print(f"Record: {df['time'].iloc[0]} to {df['time'].iloc[-1]} ({len(df)} rows)")

df = add_segment_id(df)
seg = (df.groupby("segment_id").agg(start_time=("time", "first"), end_time=("time", "last"),
                                     sample_count=("time", "size")).reset_index())
nominal_n = int(seg["sample_count"].value_counts().index[0])
minimum_n = int(round(nominal_n * minimum_block_fraction))
blocks = []
for sid, s in df.groupby("segment_id", sort=True):
    for b in split_blocks(s.reset_index(drop=True), nominal_n, minimum_n):
        b["source_segment_id"] = sid
        blocks.append(b)
print(f"Nominal burst: {nominal_n / fs / 60:.2f} min; analysis blocks: {len(blocks)}")

# -------------------------- PROCESS BURSTS --------------------------
stats_records, spectral_records, directional_records = [], [], []
interp_n = max(1, int(round(maximum_interpolation_gap_seconds * fs)))

nan_results = {k: np.nan for k in [
    "cross_ig_variance_m2_s2", "along_ig_variance_m2_s2", "ig_uv_covariance_m2_s2",
    "total_horizontal_ig_variance_m2_s2", "cross_ss_variance_m2_s2",
    "along_ss_variance_m2_s2", "ss_uv_covariance_m2_s2",
    "total_horizontal_ss_variance_m2_s2", "cross_ig_rms_m_s", "along_ig_rms_m_s",
    "horizontal_ig_rms_m_s", "cross_ss_rms_m_s", "along_ss_rms_m_s",
    "horizontal_ss_rms_m_s",
]}

for block_no, block in enumerate(blocks):
    start, end = block["time"].iloc[[0, -1]]
    mid = start + (end - start) / 2
    sid = int(block["source_segment_id"].iloc[0])
    sub = int(block["subblock_number"].iloc[0])
    n = len(block)

    for c in cells:
        ucol, vcol = c["cross_shore"], c["alongshore"]
        valid_before = block[[ucol, vcol]].notna().all(axis=1).mean()
        base = dict(
            analysis_block_number=block_no, source_segment_id=sid, subblock_number=sub,
            cell_number=c["cell_number"], height_relative_to_frame_bottom_m=c["z_bin_m"],
            start_time=start, end_time=end, mid_time=mid, sample_count=n,
            duration_minutes=n / fs / 60, valid_fraction_before_interpolation=valid_before,
        )

        reason = ""
        if n < minimum_n:
            reason = "analysis block too short"
        elif valid_before < minimum_valid_fraction:
            reason = "too many missing velocity samples"

        if reason:
            stats_records.append({**base, "accepted": False, "rejection_reason": reason,
                                  "valid_fraction_after_interpolation": np.nan, **nan_results})
            continue

        us = interp_short(block[ucol], interp_n)
        vs = interp_short(block[vcol], interp_n)
        finite = us.notna() & vs.notna()
        valid_after = finite.mean()
        if not finite.all():
            stats_records.append({**base, "accepted": False,
                                  "rejection_reason": "missing values remain after interpolation",
                                  "valid_fraction_after_interpolation": valid_after, **nan_results})
            continue

        u0, v0 = us.to_numpy(float), vs.to_numpy(float)
        if (not np.isfinite([np.std(u0), np.std(v0)]).all()
                or np.std(u0) < minimum_velocity_std_m_s
                or np.std(v0) < minimum_velocity_std_m_s):
            stats_records.append({**base, "accepted": False,
                                  "rejection_reason": "frozen or near-constant velocity signal",
                                  "valid_fraction_after_interpolation": valid_after, **nan_results})
            continue

        u, v = poly_detrend(u0, detrend_order), poly_detrend(v0, detrend_order)
        f, Suu, Svv, Suv = spectra_uv(u, v, fs)
        ss_hi = min(ss_high_hz, 0.5 * fs)

        uig = band_integral(f, Suu, ig_low_hz, ig_high_hz)
        vig = band_integral(f, Svv, ig_low_hz, ig_high_hz)
        uss = band_integral(f, Suu, ss_low_hz, ss_hi)
        vss = band_integral(f, Svv, ss_low_hz, ss_hi)
        ig = (f >= ig_low_hz) & (f < ig_high_hz)
        ss = (f >= ss_low_hz) & (f < ss_hi)
        cig = np.trapezoid(np.real(Suv[ig]), f[ig]) if ig.sum() >= 2 else np.nan
        css = np.trapezoid(np.real(Suv[ss]), f[ss]) if ss.sum() >= 2 else np.nan

        variances = np.array([uig, vig, uss, vss])
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            stats_records.append({**base, "accepted": False,
                                  "rejection_reason": "invalid or zero spectral variance",
                                  "valid_fraction_after_interpolation": valid_after, **nan_results})
            continue

        ui, vi, us, vs = np.sqrt([uig, vig, uss, vss])
        results = dict(
            cross_ig_variance_m2_s2=uig, along_ig_variance_m2_s2=vig,
            ig_uv_covariance_m2_s2=cig, total_horizontal_ig_variance_m2_s2=uig + vig,
            cross_ss_variance_m2_s2=uss, along_ss_variance_m2_s2=vss,
            ss_uv_covariance_m2_s2=css, total_horizontal_ss_variance_m2_s2=uss + vss,
            cross_ig_rms_m_s=ui, along_ig_rms_m_s=vi, horizontal_ig_rms_m_s=np.hypot(ui, vi),
            cross_ss_rms_m_s=us, along_ss_rms_m_s=vs, horizontal_ss_rms_m_s=np.hypot(us, vs),
        )
        stats_records.append({**base, "accepted": True, "rejection_reason": "",
                              "valid_fraction_after_interpolation": valid_after, **results})

        directional_records.append(pd.DataFrame({
            "case_id": case_id, "case_name": case_label, "analysis_block_number": block_no,
            "source_segment_id": sid, "subblock_number": sub, "cell_number": c["cell_number"],
            "height_relative_to_frame_bottom_m": c["z_bin_m"], "time": block["time"].to_numpy(),
            "cross_velocity_detrended_m_s": u, "along_velocity_detrended_m_s": v,
        }))
        spectral_records.append(pd.DataFrame({
            "analysis_block_number": block_no, "source_segment_id": sid,
            "subblock_number": sub, "cell_number": c["cell_number"],
            "height_relative_to_frame_bottom_m": c["z_bin_m"], "start_time": start,
            "end_time": end, "frequency_hz": f,
            "cross_psd_m2_s2_hz": Suu, "along_psd_m2_s2_hz": Svv,
        }))

statistics = pd.DataFrame(stats_records)
accepted = statistics.loc[statistics["accepted"]].sort_values(["cell_number", "mid_time"]).copy()
if accepted.empty:
    raise RuntimeError("No selected cell/burst combinations passed QC.")

spectra = pd.concat(spectral_records, ignore_index=True) if spectral_records else pd.DataFrame()
directional = pd.concat(directional_records, ignore_index=True) if directional_records else pd.DataFrame()
print("\nAccepted bursts by cell:")
print(accepted.groupby(["cell_number", "height_relative_to_frame_bottom_m"]).size())

# ----------------------------- SAVE -----------------------------
direction_file = output_folder / "all_cases_ADCP_directional_velocity.parquet"
if direction_file.exists():
    old = pd.read_parquet(direction_file)
    old = old.loc[old["case_id"] != case_id]
    directional = pd.concat([old, directional], ignore_index=True)
directional.sort_values(["case_id", "time", "cell_number"]).to_parquet(direction_file, index=False)
print(f"\nSaved directional velocity:\n{direction_file}")

burst_file = output_folder / "all_cases_ADCP_current_wave_axis.csv"  # retained filename for compatibility
save_cols = [
    "analysis_block_number", "source_segment_id", "subblock_number", "start_time", "end_time",
    "mid_time", "cell_number", "height_relative_to_frame_bottom_m",
    "cross_ig_variance_m2_s2", "along_ig_variance_m2_s2", "ig_uv_covariance_m2_s2",
    "total_horizontal_ig_variance_m2_s2", "cross_ig_rms_m_s", "along_ig_rms_m_s",
    "horizontal_ig_rms_m_s", "cross_ss_variance_m2_s2", "along_ss_variance_m2_s2",
    "ss_uv_covariance_m2_s2", "total_horizontal_ss_variance_m2_s2", "cross_ss_rms_m_s",
    "along_ss_rms_m_s", "horizontal_ss_rms_m_s",
]
burst = accepted[save_cols].copy()
burst.insert(0, "case_name", case_label)
burst.insert(0, "case_id", case_id)
for k, val in {
    "ig_low_hz": ig_low_hz, "ig_high_hz": ig_high_hz, "ss_low_hz": ss_low_hz,
    "ss_high_hz": ss_high_hz, "detrend_order": detrend_order,
    "welch_segment_seconds": welch_segment_seconds, "overlap_fraction": overlap_fraction,
}.items():
    burst[k] = val

if burst_file.exists():
    old = pd.read_csv(burst_file, parse_dates=["start_time", "end_time", "mid_time"],)

    # Remove the old results for the case being processed.
    # Results from the other cases are retained.
    old = old.loc[old["case_id"] != case_id]

    burst = pd.concat([old, burst], ignore_index=True,)

burst = burst.sort_values(["case_id", "mid_time", "cell_number"])
burst.to_csv(burst_file,index=False,)
print(f"Saved burst spectral/RMS statistics:\n{burst_file}")

# ----------------------------- PLOTS -----------------------------
# ---------------- MEAN / MEDIAN BURST AUTOSPECTRA ----------------
# Suu, Svv and Suv have already been calculated for every accepted burst
# in the processing loop above. The dataframe `spectra` stores the
# burst-resolved Suu and Svv on the Welch frequency grid, so this block
# only aggregates those already-computed burst spectra across time.

if not spectra.empty:

    def plot_aggregate_velocity_spectra(statistic):
        """Plot mean or median of the already-computed burst autospectra."""
        statistic = statistic.lower()
        if statistic not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'")

        fig, axes = plt.subplots(
            2, 2, figsize=(12, 8), sharex=True, sharey=True
        )
        axes = axes.ravel()

        legend_handles = None
        legend_labels = None

        for ax, c in zip(axes, cells):
            s = spectra.loc[
                spectra["cell_number"] == c["cell_number"]
            ].copy()

            if s.empty:
                ax.set_visible(False)
                continue

            grouped = s.groupby("frequency_hz", sort=True)

            if statistic == "mean":
                agg = grouped[[
                    "cross_psd_m2_s2_hz",
                    "along_psd_m2_s2_hz",
                ]].mean()

                f = agg.index.to_numpy(float)
                Suu = agg["cross_psd_m2_s2_hz"].to_numpy(float)
                Svv = agg["along_psd_m2_s2_hz"].to_numpy(float)

            else:
                med = grouped[[
                    "cross_psd_m2_s2_hz",
                    "along_psd_m2_s2_hz",
                ]].median()
                q25 = grouped[[
                    "cross_psd_m2_s2_hz",
                    "along_psd_m2_s2_hz",
                ]].quantile(0.25)
                q75 = grouped[[
                    "cross_psd_m2_s2_hz",
                    "along_psd_m2_s2_hz",
                ]].quantile(0.75)

                f = med.index.to_numpy(float)
                Suu = med["cross_psd_m2_s2_hz"].to_numpy(float)
                Svv = med["along_psd_m2_s2_hz"].to_numpy(float)

                # Interquartile spread across bursts
                ax.fill_between(
                    f,
                    q25["cross_psd_m2_s2_hz"].to_numpy(float),
                    q75["cross_psd_m2_s2_hz"].to_numpy(float),
                    alpha=0.12,
                )
                ax.fill_between(
                    f,
                    q25["along_psd_m2_s2_hz"].to_numpy(float),
                    q75["along_psd_m2_s2_hz"].to_numpy(float),
                    alpha=0.12,
                )

            ig_patch = ax.axvspan(
                ig_low_hz, ig_high_hz, alpha=0.16, label="IG band"
            )
            ss_patch = ax.axvspan(
                ss_low_hz, min(ss_high_hz, 0.5 * fs),
                alpha=0.10, label="Sea-swell band"
            )

            line_u, = ax.plot(
                f, Suu, linewidth=1.4,
                label=f"Cross-shore {statistic}"
            )
            line_v, = ax.plot(
                f, Svv, linewidth=1.4,
                label=f"Alongshore {statistic}"
            )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
            ax.set_title(
                f"Cell {c['cell_number']}: z={c['z_bin_m']:.3f} m"
            )
            ax.grid(True, which="both", alpha=0.22)

            if legend_handles is None:
                legend_handles = [line_u, line_v, ig_patch, ss_patch]
                legend_labels = [
                    f"Cross-shore {statistic}",
                    f"Alongshore {statistic}",
                    "IG band",
                    "Sea-swell band",
                ]

        fig.suptitle(
            f"{statistic.capitalize()} ADCP-HR velocity autospectra "
            f"({case_label})"
        )

        if legend_handles is not None:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, 0.955),
            )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()


    plot_aggregate_velocity_spectra("mean")
    plot_aggregate_velocity_spectra("median")
    
def plot_vertical_rms(column, title, ylabel):
    fig, axes = plt.subplots(len(cells), 1, figsize=(12, 2.2 * len(cells)), sharex=True)
    axes = np.atleast_1d(axes)
    ymax = accepted[column].max()
    for ax, c in zip(axes, cells):
        d = accepted.loc[accepted["cell_number"] == c["cell_number"]].sort_values("mid_time").copy()
        ax.plot(d["mid_time"], gap_plot_series(d, column))
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05 * ymax if np.isfinite(ymax) else None)
        ax.grid(True, alpha=0.25)
        ax.text(0.01, 0.88, f"Cell {c['cell_number']}, z={c['z_bin_m']:.3f} m",
                transform=ax.transAxes, va="top")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{title}\n{case_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_vertical_rms("horizontal_ig_rms_m_s", "Vertical consistency of IG velocity RMS",
                  r"$|\mathbf{U}_{\mathrm{RMS,IG}}|$\n(m/s)")
plot_vertical_rms("horizontal_ss_rms_m_s", "Vertical consistency of sea-swell velocity RMS",
                  r"$|\mathbf{U}_{\mathrm{RMS,SS}}|$\n(m/s)")

# IG/SS variance ratio at highest selected cell
highest = max(cells, key=lambda c: c["z_bin_m"])
d = accepted.loc[accepted["cell_number"] == highest["cell_number"]].sort_values("mid_time").copy()
d["R_u"] = d["horizontal_ig_rms_m_s"]**2 / d["horizontal_ss_rms_m_s"]**2
d.loc[~np.isfinite(d["R_u"]), "R_u"] = np.nan
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(d["mid_time"], gap_plot_series(d, "R_u"))
ax.set_ylabel(r"$R_u=U_{h,\mathrm{RMS,IG}}^2/U_{h,\mathrm{RMS,SS}}^2$")
ax.set_xlabel("Time")
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25)
ax.set_title(f"IG/SS velocity-variance ratio — Cell {highest['cell_number']}, z={highest['z_bin_m']:.3f} m ({case_label})")
fig.tight_layout()
plt.show()


