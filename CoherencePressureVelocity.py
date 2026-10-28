# -*- coding: utf-8 -*-
"""
ADCP-pressure / ADCP-velocity cross-spectral analysis for F1 and F3.

For 8 selected cells per frame (4 ADCP-UP + 4 ADCP-HR), calculate
frequency-dependent pressure--cross-shore and pressure--alongshore:
    - autospectra Spp, Suu, Svv
    - cross-spectra Spu, Spv
    - magnitude-squared coherence gamma^2_pu, gamma^2_pv
    - phase arg(Spu), arg(Spv)

Pressure comes from the ADCP-UP pressure parquet. Velocity comes from both
ADCP-UP and ADCP-HR velocity parquets. Principal-axis calculations are omitted.
Case IDs are matched loosely: case_id only has to contain F1 or F3.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
from scipy.signal import welch, csd

# ============================================================
# USER SETTINGS
# ============================================================
spectra_folder = Path(r"C:\dev\Python\LongWaveAnalysis\Spectra")

velocity_up_file = spectra_folder / "all_cases_ADCP_UP_directional_velocity.parquet"
velocity_hr_file = spectra_folder / "all_cases_ADCP_HR_directional_velocity.parquet"
pressure_file = spectra_folder / "all_cases_ADCP_UP_directional_pressure.parquet"

output_folder = spectra_folder
burst_output_file = output_folder / "all_cases_ADCPpressure_ADCPvelocity_IG_diagnostics_8cells.csv"
spectral_output_file = output_folder / "all_cases_ADCPpressure_ADCPvelocity_cross_spectra_8cells.parquet"

# Recent-analysis IG definition
ig_low_hz, ig_high_hz = 0.005, 0.040
welch_segment_seconds = 512.0
overlap_fraction = 0.5
minimum_samples = 300
minimum_overlap_fraction = 0.95

# Four UP + four HR cells used in the recent vertical-structure analysis.
selected_cells = {
    "F1": {
        "ADCP-UP": {19: 17.897, 14: 13.897, 6: 7.497, 1: 3.497},
        "ADCP-HR": {1: 0.403, 5: 0.283, 9: 0.163, 13: 0.043},
    },
    "F3": {
        "ADCP-UP": {14: 9.547, 11: 8.047, 6: 5.547, 1: 3.047},
        "ADCP-HR": {1: 0.394, 5: 0.274, 9: 0.154, 13: 0.034},
    },
}

# ============================================================
# HELPERS
# ============================================================
def utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def fs_from_time(t):
    x = pd.Series(t).astype("int64").to_numpy(np.float64) / 1e9
    dt = np.diff(x)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return np.nan
    q = np.nanpercentile(dt, 25)
    dt = dt[dt <= 1.5 * q] if np.any(dt <= 1.5 * q) else dt
    return 1.0 / np.nanmedian(dt)


def spectral_settings(fs, n):
    nperseg = int(round(welch_segment_seconds * fs))
    if n < nperseg or nperseg < 16:
        return None
    return nperseg, min(int(round(overlap_fraction * nperseg)), nperseg - 1)


def clean_series(d, cols):
    d = d.copy()
    d["time"] = utc(d["time"])
    return (d.dropna(subset=["time", *cols]).sort_values("time")
             .drop_duplicates("time").reset_index(drop=True))


def frame_filter(frame):
    # Arrow substring matching is avoided because case naming has varied.
    # Time/cell are filtered in Arrow; frame is filtered after the small read.
    return frame


def keep_frame(d, frame):
    return d.loc[d["case_id"].astype(str).str.contains(
        frame, case=False, na=False, regex=False)].copy()


def sync_pressure_velocity(p, v):
    """Interpolate the faster record to the slower record's timestamps."""
    p = clean_series(p, ["pressure_head_detrended_m"])
    v = clean_series(v, ["cross_velocity_detrended_m_s", "along_velocity_detrended_m_s"])
    if len(p) < minimum_samples or len(v) < minimum_samples:
        return None

    tp = p["time"].astype("int64").to_numpy(np.float64) / 1e9
    tv = v["time"].astype("int64").to_numpy(np.float64) / 1e9
    pp = p["pressure_head_detrended_m"].to_numpy(float)
    uu = v["cross_velocity_detrended_m_s"].to_numpy(float)
    vv = v["along_velocity_detrended_m_s"].to_numpy(float)
    fp, fv = fs_from_time(p["time"]), fs_from_time(v["time"])
    if not (np.isfinite(fp) and np.isfinite(fv) and fp > 0 and fv > 0):
        return None

    start, end = max(tp.min(), tv.min()), min(tp.max(), tv.max())
    if end <= start:
        return None

    if fp <= fv:  # pressure is slower: use pressure clock
        m = (tp >= start) & (tp <= end)
        t, pressure = tp[m], pp[m]
        u, along = np.interp(t, tv, uu), np.interp(t, tv, vv)
        fs = fp
        overlap = m.mean()
    else:         # velocity is slower: use velocity clock
        m = (tv >= start) & (tv <= end)
        t, u, along = tv[m], uu[m], vv[m]
        pressure = np.interp(t, tp, pp)
        fs = fv
        overlap = m.mean()

    good = np.isfinite(pressure) & np.isfinite(u) & np.isfinite(along)
    if good.sum() < minimum_samples or overlap < minimum_overlap_fraction:
        return None
    return t[good], pressure[good], u[good], along[good], fs, overlap


def cross_spectra(p, u, v, fs):
    settings = spectral_settings(fs, len(p))
    if settings is None:
        return None
    nperseg, noverlap = settings
    kw = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
              detrend=False, scaling="density")
    f, spp = welch(p, **kw)
    _, suu = welch(u, **kw)
    _, svv = welch(v, **kw)
    _, spu = csd(p, u, **kw)  # conj(P)*U: velocity phase relative to pressure
    _, spv = csd(p, v, **kw)
    cpu = np.abs(spu)**2 / (spp * suu)
    cpv = np.abs(spv)**2 / (spp * svv)
    cpu = np.clip(cpu, 0, 1)
    cpv = np.clip(cpv, 0, 1)
    return f, spp, suu, svv, spu, spv, cpu, cpv


def band_summary(f, coh, cross):
    m = ((f >= ig_low_hz) & (f < ig_high_hz) & np.isfinite(coh)
         & np.isfinite(cross))
    if m.sum() < 2:
        return np.nan, np.nan, 0
    # Coherence: median across IG bins. Phase: angle of integrated complex CSD.
    med_coh = float(np.nanmedian(coh[m]))
    phase = float(np.degrees(np.angle(np.trapezoid(cross[m], f[m]))))
    return med_coh, phase, int(m.sum())


def pressure_burst_index(dataset):
    """Build case/block start-end index batchwise, without loading pressure data at once."""
    need = {"case_id", "analysis_block_number", "time"}
    if not need.issubset(dataset.schema.names):
        raise KeyError(f"Pressure parquet missing: {sorted(need-set(dataset.schema.names))}")
    parts = []
    for batch in dataset.scanner(columns=list(need), batch_size=250_000).to_batches():
        d = batch.to_pandas()
        d["time"] = utc(d["time"])
        d = d.dropna(subset=["case_id", "analysis_block_number", "time"])
        if len(d):
            parts.append(d.groupby(["case_id", "analysis_block_number"], as_index=False)
                         .agg(start_time=("time", "min"), end_time=("time", "max")))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return (out.groupby(["case_id", "analysis_block_number"], as_index=False)
            .agg(start_time=("start_time", "min"), end_time=("end_time", "max")))


# ============================================================
# OPEN DATASETS + CHECK SCHEMAS
# ============================================================
pds = ds.dataset(pressure_file, format="parquet")
vsets = {
    "ADCP-UP": ds.dataset(velocity_up_file, format="parquet"),
    "ADCP-HR": ds.dataset(velocity_hr_file, format="parquet"),
}

pcols = ["case_id", "analysis_block_number", "time", "pressure_head_detrended_m"]
vcols = ["case_id", "cell_number", "height_relative_to_frame_bottom_m", "time",
         "cross_velocity_detrended_m_s", "along_velocity_detrended_m_s"]

for c in pcols:
    if c not in pds.schema.names:
        raise KeyError(f"Pressure parquet missing column: {c}")
for instrument, dataset in vsets.items():
    missing = set(vcols) - set(dataset.schema.names)
    if missing:
        raise KeyError(f"{instrument} velocity parquet missing: {sorted(missing)}")

print("Building ADCP-pressure burst index ...")
bursts = pressure_burst_index(pds)
bursts["frame_id"] = bursts["case_id"].astype(str).str.extract(r"(F1|F3)", expand=False)
bursts = bursts.loc[bursts["frame_id"].isin(selected_cells)].sort_values("start_time")
print(bursts.groupby("frame_id").size())

# ============================================================
# BURST-BY-BURST ANALYSIS
# ============================================================
burst_records, spectral_records = [], []

for ib, b in bursts.iterrows():
    frame = b["frame_id"]
    block = int(b["analysis_block_number"])
    t0, t1 = b["start_time"], b["end_time"]
    t0p, t1p = t0.to_pydatetime(), t1.to_pydatetime()

    pfilt = ((ds.field("analysis_block_number") == block)
             & (ds.field("time") >= t0p) & (ds.field("time") <= t1p))
    p = pds.to_table(columns=pcols, filter=pfilt).to_pandas()
    p = keep_frame(p, frame)
    if p.empty:
        continue
    # If block numbers repeat across pressure cases, retain the burst's case.
    p = p.loc[p["case_id"].astype(str) == str(b["case_id"])].copy()
    if len(p) < minimum_samples:
        continue

    for instrument, cells in selected_cells[frame].items():
        dataset = vsets[instrument]
        cell_list = list(cells)
        vfilt = ((ds.field("cell_number").isin(cell_list))
                 & (ds.field("time") >= t0p) & (ds.field("time") <= t1p))
        vel = dataset.to_table(columns=vcols, filter=vfilt).to_pandas()
        vel = keep_frame(vel, frame)
        if vel.empty:
            continue

        for cell in cell_list:
            vc = vel.loc[vel["cell_number"] == cell].copy()
            if len(vc) < minimum_samples:
                continue
            synced = sync_pressure_velocity(p, vc)
            if synced is None:
                continue
            _, pressure, u, v, fs, overlap = synced
            spec = cross_spectra(pressure, u, v, fs)
            if spec is None:
                continue
            f, spp, suu, svv, spu, spv, cpu, cpv = spec
            coh_u, phase_u, bins_u = band_summary(f, cpu, spu)
            coh_v, phase_v, bins_v = band_summary(f, cpv, spv)
            height = float(np.nanmedian(vc["height_relative_to_frame_bottom_m"]))
            mid = t0 + (t1 - t0) / 2

            burst_records.append(dict(
                frame_id=frame, instrument=instrument, cell_number=int(cell),
                height_relative_to_frame_bottom_m=height,
                pressure_case_id=str(b["case_id"]), analysis_block_number=block,
                start_time=t0, end_time=t1, mid_time=mid,
                sampling_frequency_hz=fs, matched_sample_count=len(pressure),
                overlap_fraction=overlap, ig_low_hz=ig_low_hz, ig_high_hz=ig_high_hz,
                ig_coherence_pu_median=coh_u, ig_phase_pu_deg=phase_u,
                ig_coherence_pv_median=coh_v, ig_phase_pv_deg=phase_v,
                ig_frequency_bins=min(bins_u, bins_v),
            ))

            spectral_records.append(pd.DataFrame({
                "frame_id": frame, "instrument": instrument, "cell_number": int(cell),
                "height_relative_to_frame_bottom_m": height,
                "pressure_case_id": str(b["case_id"]), "analysis_block_number": block,
                "start_time": t0, "end_time": t1, "mid_time": mid,
                "frequency_hz": f,
                "Spp_m2_hz": spp, "Suu_m2_s2_hz": suu, "Svv_m2_s2_hz": svv,
                "Spu_real": np.real(spu), "Spu_imag": np.imag(spu),
                "Spv_real": np.real(spv), "Spv_imag": np.imag(spv),
                "coherence_pu": cpu, "coherence_pv": cpv,
                "phase_pu_deg": np.degrees(np.angle(spu)),
                "phase_pv_deg": np.degrees(np.angle(spv)),
            }))

    if (ib + 1) % 25 == 0:
        print(f"Processed {ib + 1}/{len(bursts)} pressure bursts")

# # ============================================================
# # SAVE
# # ============================================================
diagnostics = pd.DataFrame(burst_records)
spectra = pd.concat(spectral_records, ignore_index=True) if spectral_records else pd.DataFrame()
# if diagnostics.empty or spectra.empty:
#     raise RuntimeError("No pressure/velocity bursts passed the matching and QC criteria.")

# diagnostics.to_csv(burst_output_file, index=False)
# spectra.to_parquet(spectral_output_file, index=False)
# print(f"\nSaved:\n{burst_output_file}\n{spectral_output_file}")
# print("\nAccepted burst counts:")
# print(diagnostics.groupby(["frame_id", "instrument", "cell_number"]).size())

# ============================================================
# DEPLOYMENT-MEDIAN FREQUENCY-DOMAIN COHERENCE + IQR
# 8 cells per frame
#
# Line  = median coherence across bursts
# Shade = interquartile range (25th--75th percentile)
# ============================================================

spectra["frequency_bin_hz"] = spectra["frequency_hz"].round(5)

for frame in ["F1", "F3"]:

    dframe = spectra.loc[
        (spectra["frame_id"] == frame)
        & (spectra["frequency_hz"] >= ig_low_hz)
        & (spectra["frequency_hz"] <= ig_high_hz)
    ].copy()

    # Preserve desired vertical order:
    # UP high -> low, then HR high -> low
    order = [
        (inst, cell, z)
        for inst, cells in selected_cells[frame].items()
        for cell, z in cells.items()
    ]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 7),
        sharex=True,
        sharey=True
    )

    for ax, (instrument, cell, z_nom) in zip(axes.flat, order):

        d = dframe.loc[
            (dframe["instrument"] == instrument)
            & (dframe["cell_number"] == cell)
        ].copy()

        if d.empty:
            ax.set_title(f"{instrument} cell {cell}\nNO DATA")
            continue

        # ----------------------------------------------------
        # Statistics across bursts at each frequency
        # ----------------------------------------------------
        s = (
            d.groupby("frequency_bin_hz", as_index=False)
            .agg(
                # p-u coherence
                coh_u=("coherence_pu", "median"),
                coh_u_q25=("coherence_pu",
                           lambda x: np.nanpercentile(x, 25)),
                coh_u_q75=("coherence_pu",
                           lambda x: np.nanpercentile(x, 75)),

                # p-v coherence
                coh_v=("coherence_pv", "median"),
                coh_v_q25=("coherence_pv",
                           lambda x: np.nanpercentile(x, 25)),
                coh_v_q75=("coherence_pv",
                           lambda x: np.nanpercentile(x, 75)),

                # Number of contributing bursts
                n=("analysis_block_number", "nunique"),

                # Actual measurement height
                z=("height_relative_to_frame_bottom_m", "median"),
            )
            .sort_values("frequency_bin_hz")
        )

        f = s["frequency_bin_hz"].to_numpy()

        cu = s["coh_u"].to_numpy()
        cu25 = s["coh_u_q25"].to_numpy()
        cu75 = s["coh_u_q75"].to_numpy()

        cv = s["coh_v"].to_numpy()
        cv25 = s["coh_v_q25"].to_numpy()
        cv75 = s["coh_v_q75"].to_numpy()

        z = float(s["z"].median())
        n = int(s["n"].max())

        # ----------------------------------------------------
        # Median coherence curves
        # ----------------------------------------------------
        line_u, = ax.plot(
            f,
            cu,
            lw=1.6,
            label=r"$p-u$ median"
        )

        line_v, = ax.plot(
            f,
            cv,
            "--",
            lw=1.6,
            label=r"$p-v$ median"
        )

        # ----------------------------------------------------
        # Interquartile range across bursts
        # ----------------------------------------------------
        ax.fill_between(
            f,
            cu25,
            cu75,
            color=line_u.get_color(),
            alpha=0.18,
            linewidth=0
        )

        ax.fill_between(
            f,
            cv25,
            cv75,
            color=line_v.get_color(),
            alpha=0.18,
            linewidth=0
        )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------
        ax.set_xlim(ig_low_hz, ig_high_hz)
        ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.25)

        ax.set_title(
            f"z={z:.2f} m | {instrument}\n"
            f"N={n}"
        )

    # Axis labels
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"Coherence $\gamma^2$")

    axes.flat[0].legend()

    fig.suptitle(
        f"{frame}: ADCP pressure–velocity coherence, IG band\n"
        "median and interquartile range across bursts"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    
# ============================================================
# SELECTED PERIODS: FULL-FREQUENCY COHERENCE + CROSS-SPECTRAL PHASE
# Append after the existing plotting block (requires spectra,
# selected_cells, output_folder, ig_high_hz, np, pd, plt).
# ============================================================
# ============================================================
# SELECTED PERIODS: COHERENCE + CIRCULAR PRESSURE–VELOCITY PHASE
# Requires: spectra, selected_cells, ig_high_hz, np, pd, plt
# ============================================================
period_year = 2018
periods = {
    'F1': [('Apr10_Apr13', f'{period_year}-04-10', f'{period_year}-04-13'),
           ('Apr29_May02', f'{period_year}-04-29', f'{period_year}-05-02')],
    'F3': [('Apr05_Apr08', f'{period_year}-04-05', f'{period_year}-04-08'),
           ('Apr10_Apr13', f'{period_year}-04-10', f'{period_year}-04-13')],
}
plot_fmax_hz = 0.20
fit_low_hz, fit_high_hz = 0.08, 0.18
min_median_coherence = 0.50
min_fit_bins = 8
min_valid_hr_cells = 2

spectra['mid_time'] = pd.to_datetime(spectra['mid_time'], utc=True)
spectra['frequency_bin_hz'] = spectra['frequency_hz'].round(5)


def period_cell_spectrum(d):
    """Aggregate bursts at each frequency, retaining complex cross-spectra."""
    return (d.groupby('frequency_bin_hz', sort=True)
             .agg(cu=('coherence_pu', 'median'),
                  cu25=('coherence_pu', lambda x: np.nanpercentile(x, 25)),
                  cu75=('coherence_pu', lambda x: np.nanpercentile(x, 75)),
                  cv=('coherence_pv', 'median'),
                  cv25=('coherence_pv', lambda x: np.nanpercentile(x, 25)),
                  cv75=('coherence_pv', lambda x: np.nanpercentile(x, 75)),
                  spu_re=('Spu_real', 'mean'),
                  spu_im=('Spu_imag', 'mean'),
                  spv_re=('Spv_real', 'mean'),
                  spv_im=('Spv_imag', 'mean'),
                  z=('height_relative_to_frame_bottom_m', 'median'))
             .reset_index())


def apparent_delay_from_pu(s):
    """Estimate apparent p-u phase slope from a contiguous SS interval."""
    f = s['frequency_bin_hz'].to_numpy(dtype=float)
    spu = s['spu_re'].to_numpy() + 1j * s['spu_im'].to_numpy()
    coh = s['cu'].to_numpy(dtype=float)
    band = ((f >= fit_low_hz) & (f <= fit_high_hz)
            & np.isfinite(spu) & (np.abs(spu) > 0)
            & np.isfinite(coh))
    if band.sum() < min_fit_bins:
        return np.nan
    # Check the band's median coherence, rather than dropping individual bins.
    if np.nanmedian(coh[band]) < min_median_coherence:
        return np.nan
    f_fit = f[band]
    phase_fit = np.unwrap(np.angle(spu[band]))
    slope, _ = np.polyfit(f_fit, phase_fit, 1)
    return float(-slope / (2 * np.pi))


for frame, frame_periods in periods.items():
    order = [(inst, cell, z)
             for inst, cells in selected_cells[frame].items()
             for cell, z in cells.items()]

    for tag, start_date, end_date in frame_periods:
        start = pd.Timestamp(start_date, tz='UTC')
        end_exclusive = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
        period_data = spectra.loc[
            (spectra['frame_id'] == frame)
            & (spectra['mid_time'] >= start)
            & (spectra['mid_time'] < end_exclusive)
            & (spectra['frequency_hz'] > 0)
        ].copy()
        if plot_fmax_hz is not None:
            period_data = period_data.loc[
                period_data['frequency_hz'] <= plot_fmax_hz]
        if period_data.empty:
            print(f'No spectral data: {frame}, {tag}')
            continue

        # Build every cell's period-averaged spectrum before estimating delay.
        cell_spectra = {}
        for inst, cell, _ in order:
            d = period_data.loc[
                (period_data['instrument'] == inst)
                & (period_data['cell_number'] == cell)]
            if not d.empty:
                cell_spectra[(inst, cell)] = period_cell_spectrum(d)

        # Estimate one shared apparent HR delay for this frame and period.
        hr_delays = []
        for inst, cell, _ in order:
            if inst != 'ADCP-HR' or (inst, cell) not in cell_spectra:
                continue
            tau = apparent_delay_from_pu(cell_spectra[(inst, cell)])
            print(f'{frame} | {tag} | HR cell {cell} | '
                  f'apparent delay = {tau:.2f} s' if np.isfinite(tau)
                  else f'{frame} | {tag} | HR cell {cell} | no valid delay')
            if np.isfinite(tau):
                hr_delays.append(tau)

        tau_period = (float(np.median(hr_delays))
                      if len(hr_delays) >= min_valid_hr_cells else np.nan)
        status = (f'HR adjusted: {tau_period:+.2f} s (apparent)'
                  if np.isfinite(tau_period) else 'HR uncorrected: no reliable delay')
        print(f'{frame} | {tag} | {status}')

        fig, axes = plt.subplots(2, len(order), figsize=(22, 7.5),
                                 sharex=True, sharey='row', squeeze=False)
        for j, (inst, cell, z_nom) in enumerate(order):
            axc, axp = axes[:, j]
            s = cell_spectra.get((inst, cell))
            if s is None:
                axc.set_title(f'{inst} cell {cell}\nNO DATA', fontsize=10)
                continue

            f = s['frequency_bin_hz'].to_numpy(dtype=float)
            spu = s['spu_re'].to_numpy() + 1j * s['spu_im'].to_numpy()
            spv = s['spv_re'].to_numpy() + 1j * s['spv_im'].to_numpy()

            # Correct HR cross-spectral phase only; UP is always original.
            if inst == 'ADCP-HR' and np.isfinite(tau_period):
                phase_factor = np.exp(1j * 2 * np.pi * f * tau_period)
                spu = spu * phase_factor
                spv = spv * phase_factor

            # Circular phase difference: velocity relative to pressure.
            # np.angle returns angles in [-pi, pi]; no UP-cell reference.
            phase_pu = np.degrees(np.angle(spu))
            # Represent positive phases using equivalent negative angles
            phase_pu = np.where(phase_pu > 0, phase_pu - 360, phase_pu)
            
            phase_pv = np.degrees(np.angle(spv))

            lu, = axc.plot(f, s['cu'], lw=1.4, label='p–u')
            lv, = axc.plot(f, s['cv'], '--', lw=1.4, label='p–v')
            axc.fill_between(f, s['cu25'], s['cu75'],
                             color=lu.get_color(), alpha=0.17)
            axc.fill_between(f, s['cv25'], s['cv75'],
                             color=lv.get_color(), alpha=0.17)
            axp.plot(f, phase_pu, color=lu.get_color(), lw=1.1)
            axp.plot(f, phase_pv, '--', color=lv.get_color(), lw=1.1)

            d = period_data.loc[
                (period_data['instrument'] == inst)
                & (period_data['cell_number'] == cell)]
            n = d[['pressure_case_id', 'analysis_block_number']].drop_duplicates().shape[0]
            phase_note = ('original' if inst == 'ADCP-UP'
                          else ('adjusted' if np.isfinite(tau_period)
                                else 'uncorrected'))
            axc.set_title(f'{inst} cell {cell}\nz={s["z"].median():.2f} m; '
                          f'N={n}\n{phase_note}', fontsize=9)

        fmax = (plot_fmax_hz if plot_fmax_hz is not None
                else period_data['frequency_hz'].max())
        for ax in axes.flat:
            ax.axvline(ig_high_hz, color='0.4', ls=':', lw=1)
            ax.set_xlim(0, fmax)
            ax.grid(alpha=0.25)
        for ax in axes[0]:
            ax.set_ylim(0, 1)
        for ax in axes[1]:
            ax.set_ylim(-360, 180)
            ax.set_yticks([-360, -180, -90, 0, 90, 180])
            ax.set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel(r'Coherence $\gamma^2$')
        axes[1, 0].set_ylabel('Circular p–velocity phase (degrees)')
        axes[0, 0].legend(loc='best', fontsize=9)
        fig.suptitle(f'{frame} | {start_date} to {end_date}\n'
                     f'Median coherence ± IQR; circular pressure–velocity phase',
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()
