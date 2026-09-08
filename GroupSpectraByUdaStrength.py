import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from scipy.signal import welch, csd

spectra_folder = Path(r"Spectra")
processed_folder = Path(r"Processed")

directional_file = spectra_folder / "all_cases_ADCP_directional_velocity.parquet"
current_file = processed_folder / "DVN_F3_ADCP_upward_depth_averaged_current.csv"

case_id = "F3"
cells_to_plot = [1, 5, 9, 13]

welch_segment_seconds = 512.0
overlap_fraction = 0.50
current_match_tolerance = pd.Timedelta("10min")

fmin_plot = 0.005
fmax_plot = 1.0
f_ig_high = 0.05


def find_time_column(df):
    for c in ["time", "datetime", "timestamp", "date_time",
              "Time", "Datetime", "Timestamp"]:
        if c in df.columns:
            return c
    raise KeyError("No time column found in current file.")


def split_into_bursts(df):
    d = df.sort_values("time").copy()
    dt = d["time"].diff().dt.total_seconds()
    good = dt[(dt > 0) & np.isfinite(dt)]
    if good.empty:
        d["burst_number"] = 0
        return d
    median_dt = good.median()
    gap_threshold = max(2.0, 5.0 * median_dt)
    d["burst_number"] = (dt.isna() | (dt > gap_threshold)).cumsum() - 1
    return d


def get_fs(time):
    dt = pd.to_datetime(time).sort_values().diff().dt.total_seconds().to_numpy()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return np.nan if len(dt) == 0 else 1.0 / np.median(dt)


def calc_psd(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    nperseg = int(round(welch_segment_seconds * fs))
    if len(x) < nperseg:
        return None, None

    noverlap = int(round(overlap_fraction * nperseg))

    return welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,   # already quadratically detrended burst by burst
        scaling="density",
    )


# ------------------------------------------------------------
# Read only needed velocity columns
# ------------------------------------------------------------
velocity = pd.read_parquet(
    directional_file,
    columns=[
        "case_id",
        "cell_number",
        "height_relative_to_frame_bottom_m",
        "time",
        "cross_velocity_detrended_m_s",
        "along_velocity_detrended_m_s",
    ],
)

velocity["time"] = pd.to_datetime(velocity["time"], errors="coerce")
velocity["cell_number"] = pd.to_numeric(velocity["cell_number"], errors="coerce")

velocity["frame"] = (
    velocity["case_id"]
    .astype(str)
    .str.upper()
    .str.extract(r"(F1|F3)", expand=False)
)

velocity = velocity.loc[
    (velocity["frame"] == case_id)
    & velocity["cell_number"].isin(cells_to_plot)
].dropna(subset=["time"]).copy()

if velocity.empty:
    raise ValueError(f"No {case_id} velocity data found.")


# ------------------------------------------------------------
# Read depth-averaged current
# ------------------------------------------------------------
current = pd.read_csv(current_file)

tcol = find_time_column(current)
current["time"] = pd.to_datetime(current[tcol], errors="coerce")

if "depth_avg_current_speed_m_s" not in current.columns:
    current["depth_avg_current_speed_m_s"] = np.hypot(
        current["depth_avg_cross_shore_m_s"],
        current["depth_avg_alongshore_m_s"],
    )

# Keep current magnitude AND direction components
current = (
    current[
        [
            "time",
            "depth_avg_cross_shore_m_s",
            "depth_avg_alongshore_m_s",
            "depth_avg_current_speed_m_s",
        ]
    ]
    .dropna()
    .sort_values("time")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Determine tercile thresholds from one reference cell
# ------------------------------------------------------------
reference = velocity.loc[velocity["cell_number"] == cells_to_plot[0]].copy()
reference = split_into_bursts(reference)

burst_times = (
    reference.groupby("burst_number")["time"]
    .agg(lambda x: x.iloc[len(x) // 2])
    .reset_index(name="burst_time")
    .sort_values("burst_time")
)

matched = pd.merge_asof(
    burst_times,
    current,
    left_on="burst_time",
    right_on="time",
    direction="nearest",
    tolerance=current_match_tolerance,
)

matched = matched.dropna(subset=["depth_avg_current_speed_m_s"])

q1, q2 = matched["depth_avg_current_speed_m_s"].quantile([1/3, 2/3])

print(
    f"{case_id} thresholds:\n"
    f"  Weak:   |U_DA| < {q1:.3f} m/s\n"
    f"  Strong: |U_DA| >= {q2:.3f} m/s\n"
    f"  Middle third ignored."
)


# # ------------------------------------------------------------
# # Compute spectra only for weak and strong bursts
# # ------------------------------------------------------------
# spectra = {
#     group: {
#         cell: {"f": None, "Suu": [], "Svv": []}
#         for cell in cells_to_plot
#     }
#     for group in ["Weak", "Strong"]
# }

# cell_heights = {}

# for cell in cells_to_plot:
#     print(f"Processing cell {cell} ...")

#     dc = velocity.loc[velocity["cell_number"] == cell].copy()
#     if dc.empty:
#         continue

#     cell_heights[cell] = dc["height_relative_to_frame_bottom_m"].median()
#     dc = split_into_bursts(dc)

#     for _, burst in dc.groupby("burst_number", sort=False):
#         burst = burst.sort_values("time")
#         if len(burst) < 10:
#             continue

#         burst_midtime = burst["time"].iloc[len(burst) // 2]

#         # nearest current value
#         pos = current["time"].searchsorted(burst_midtime)
#         candidates = []
#         for j in (pos - 1, pos):
#             if 0 <= j < len(current):
#                 dt = abs(current.iloc[j]["time"] - burst_midtime)
#                 candidates.append((dt, j))

#         if not candidates:
#             continue

#         dt, j = min(candidates, key=lambda x: x[0])
#         if dt > current_match_tolerance:
#             continue

#         Uda = current.iloc[j]["depth_avg_current_speed_m_s"]

#         if Uda < q1:
#             group = "Weak"
#         elif Uda >= q2:
#             group = "Strong"
#         else:
#             continue

#         fs = get_fs(burst["time"])
#         if not np.isfinite(fs):
#             continue

#         f, Suu = calc_psd(burst["cross_velocity_detrended_m_s"], fs)
#         _, Svv = calc_psd(burst["along_velocity_detrended_m_s"], fs)

#         if f is None or Svv is None:
#             continue

#         f0 = spectra[group][cell]["f"]

#         if f0 is None:
#             spectra[group][cell]["f"] = f
#         elif len(f0) != len(f) or not np.allclose(f0, f):
#             continue

#         spectra[group][cell]["Suu"].append(Suu)
#         spectra[group][cell]["Svv"].append(Svv)


# # ------------------------------------------------------------
# # Plot median spectra
# # rows: Suu / Svv
# # cols: weak / strong
# # four cells overlaid
# # ------------------------------------------------------------
frame_number = case_id.replace("F", "")
# fig, axes = plt.subplots(
#     2, 2,
#     figsize=(11, 8),
#     sharex=True,
#     sharey="row",
#     constrained_layout=True,
# )

# for col, group in enumerate(["Weak", "Strong"]):

#     for cell in cells_to_plot:
#         d = spectra[group][cell]

#         if d["f"] is None or len(d["Suu"]) == 0:
#             continue

#         f = d["f"]
#         Suu_med = np.nanmedian(np.vstack(d["Suu"]), axis=0)
#         Svv_med = np.nanmedian(np.vstack(d["Svv"]), axis=0)

#         z = cell_heights.get(cell, np.nan)
#         label = f"Cell {cell} (z={z:.3f} m)" if np.isfinite(z) else f"Cell {cell}"

#         axes[0, col].loglog(f, Suu_med, label=label)
#         axes[1, col].loglog(f, Svv_med, label=label)

#         print(f"{group:6s} | cell {cell:2d}: {len(d['Suu'])} bursts")

#     title = (
#         f"Weak current: |U_DA| < {q1:.2f} m/s"
#         if group == "Weak"
#         else f"Strong current: |U_DA| >= {q2:.2f} m/s"
#     )
#     axes[0, col].set_title(title)

#     for row in range(2):
#         ax = axes[row, col]
#         ax.set_xlim(fmin_plot, fmax_plot)
#         ax.axvline(f_ig_high, linestyle="--", linewidth=1)
#         ax.grid(True, which="both", alpha=0.25)

#     axes[1, col].set_xlabel("Frequency (Hz)")

# axes[0, 0].set_ylabel(r"Median $S_{uu}$ (m$^2$ s$^{-2}$ Hz$^{-1}$)")
# axes[1, 0].set_ylabel(r"Median $S_{vv}$ (m$^2$ s$^{-2}$ Hz$^{-1}$)")

# axes[0, 1].legend(loc="best", fontsize=9)

# fig.suptitle(
#     f"Frame {frame_number}: velocity spectra during weak and strong" r"$|\mathbf{U}_{DA}|$",
#     fontsize=14,
# )

# plt.show()

# # ------------------------------------------------------------
# # Plot strong-to-weak spectral amplification
# #
# # Au = median Suu,strong / median Suu,weak
# # Av = median Svv,strong / median Svv,weak
# # ------------------------------------------------------------
# fig, axes = plt.subplots(
#     1, 2,
#     figsize=(11, 4.5),
#     sharex=True,
#     constrained_layout=True,
# )

# for cell in cells_to_plot:

#     weak = spectra["Weak"][cell]
#     strong = spectra["Strong"][cell]

#     # Need spectra in both current groups
#     if (
#         weak["f"] is None
#         or strong["f"] is None
#         or len(weak["Suu"]) == 0
#         or len(strong["Suu"]) == 0
#     ):
#         continue

#     f_weak = weak["f"]
#     f_strong = strong["f"]

#     # Frequency vectors should be identical
#     if (
#         len(f_weak) != len(f_strong)
#         or not np.allclose(f_weak, f_strong)
#     ):
#         print(f"Cell {cell}: weak/strong frequency grids differ; skipping.")
#         continue

#     f = f_weak

#     # Median spectra across bursts
#     Suu_weak = np.nanmedian(
#         np.vstack(weak["Suu"]),
#         axis=0,
#     )
#     Suu_strong = np.nanmedian(
#         np.vstack(strong["Suu"]),
#         axis=0,
#     )

#     Svv_weak = np.nanmedian(
#         np.vstack(weak["Svv"]),
#         axis=0,
#     )
#     Svv_strong = np.nanmedian(
#         np.vstack(strong["Svv"]),
#         axis=0,
#     )

#     # Strong / weak amplification
#     Au = Suu_strong / Suu_weak
#     Av = Svv_strong / Svv_weak

#     z = cell_heights.get(cell, np.nan)

#     label = (
#         f"Cell {cell} (z={z:.3f} m)"
#         if np.isfinite(z)
#         else f"Cell {cell}"
#     )

#     axes[0].semilogx(
#         f,
#         Au,
#         label=label,
#     )

#     axes[1].semilogx(
#         f,
#         Av,
#         label=label,
#     )


# # ------------------------------------------------------------
# # Formatting
# # ------------------------------------------------------------
# for ax in axes:

#     ax.set_xlim(
#         fmin_plot,
#         fmax_plot,
#     )

#     # A = 1 means no difference between strong and weak current
#     ax.axhline(
#         1.0,
#         linestyle="--",
#         linewidth=1,
#     )

#     # IG / SS boundary
#     ax.axvline(
#         f_ig_high,
#         linestyle="--",
#         linewidth=1,
#     )

#     ax.grid(
#         True,
#         which="both",
#         alpha=0.25,
#     )

#     ax.set_xlabel(
#         "Frequency (Hz)"
#     )


# axes[0].set_ylabel(
#     r"$A_u(f)="
#     r"S_{uu,\mathrm{strong}}/"
#     r"S_{uu,\mathrm{weak}}$"
# )

# axes[1].set_ylabel(
#     r"$A_v(f)="
#     r"S_{vv,\mathrm{strong}}/"
#     r"S_{vv,\mathrm{weak}}$"
# )

# axes[0].set_title(
#     "Cross-shore spectral amplification"
# )

# axes[1].set_title(
#     "Alongshore spectral amplification"
# )

# axes[1].legend(
#     loc="best",
#     fontsize=9,
# )

# fig.suptitle(
#     f"Frame {frame_number}: strong-to-weak current spectral amplification",
#     fontsize=14,
# )

# plt.show()

# ============================================================
# PRINCIPAL OSCILLATION AXIS FOR ALL BURSTS
#
# Bands:
#   IG       = 0.005 - 0.05 Hz
#   SS_peak  = 0.05  - 0.20 Hz
#   SS_high  = 0.20  - 1.00 Hz
#
# For each burst and cell calculate:
#   variance u
#   variance v
#   covariance uv
#   principal-axis angle
#   ellipse anisotropy A_ellipse
#
# Angle convention:
#   0 deg  = positive cross-shore axis
#   90 deg = alongshore axis
#
# Since this is an AXIS rather than a directed vector,
# angles are mapped to 0-180 deg.
# ============================================================

bands = {
    "IG": (0.005, 0.05),
    "SS_peak": (0.05, 0.20),
    "SS_high": (0.20, 1.00),
}


def burst_uv_spectra(u, v, fs):
    """
    Calculate Suu, Svv and complex Suv using exactly the same
    samples and Welch settings.
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Important: same finite samples for u and v
    good = np.isfinite(u) & np.isfinite(v)

    u = u[good]
    v = v[good]

    nperseg = int(round(welch_segment_seconds * fs))

    if len(u) < nperseg:
        return None, None, None, None

    noverlap = int(round(overlap_fraction * nperseg))

    f, Suu = welch(
        u,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    _, Svv = welch(
        v,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    _, Suv = csd(
        u,
        v,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    return f, Suu, Svv, Suv


def integrate_band(f, S, f_low, f_high):
    """
    Integrate spectrum over one frequency band.
    """
    mask = (
        np.isfinite(f)
        & np.isfinite(S)
        & (f >= f_low)
        & (f < f_high)
    )

    if np.sum(mask) < 2:
        return np.nan

    return np.trapezoid(
    S[mask],
    f[mask],
)


def principal_axis_from_covariance(var_u, var_v, cov_uv):
    """
    Principal oscillation axis and ellipse anisotropy.

    theta:
        principal-axis orientation in degrees, 0-180.

    A_ellipse:
        0 = isotropic / direction poorly defined
        1 = strongly linear / well-defined axis
    """

    if not (
        np.isfinite(var_u)
        and np.isfinite(var_v)
        and np.isfinite(cov_uv)
    ):
        return np.nan, np.nan, np.nan, np.nan

    C = np.array(
        [
            [var_u, cov_uv],
            [cov_uv, var_v],
        ]
    )

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # np.linalg.eigh returns ascending eigenvalues
    i_max = np.argmax(eigenvalues)
    i_min = np.argmin(eigenvalues)

    lambda_max = eigenvalues[i_max]
    lambda_min = eigenvalues[i_min]

    major_vector = eigenvectors[:, i_max]

    # vector components:
    # [cross-shore, alongshore]
    theta = np.degrees(
        np.arctan2(
            major_vector[1],
            major_vector[0],
        )
    )

    # Axis has 180-deg ambiguity
    theta = theta % 180.0

    denom = lambda_max + lambda_min

    if denom > 0:
        A_ellipse = (
            (lambda_max - lambda_min)
            / denom
        )
    else:
        A_ellipse = np.nan

    return (
        theta,
        A_ellipse,
        lambda_max,
        lambda_min,
    )


# ============================================================
# LOOP THROUGH ALL BURSTS AND ALL FOUR CELLS
# ============================================================

axis_records = []

for cell in cells_to_plot:

    print(
        f"Principal-axis analysis: "
        f"{case_id}, cell {cell}"
    )

    dc = velocity.loc[
        velocity["cell_number"] == cell
    ].copy()

    if dc.empty:
        continue

    dc = split_into_bursts(dc)

    z_cell = (
        dc[
            "height_relative_to_frame_bottom_m"
        ]
        .median()
    )

    for burst_number, burst in dc.groupby(
        "burst_number",
        sort=False,
    ):

        burst = burst.sort_values("time")

        if len(burst) < 10:
            continue

        burst_midtime = (
            burst["time"]
            .iloc[len(burst) // 2]
        )

        fs = get_fs(burst["time"])

        if not np.isfinite(fs):
            continue

        # --------------------------------------------
        # Velocity spectra + cross-spectrum
        # --------------------------------------------
        f, Suu, Svv, Suv = burst_uv_spectra(
            burst[
                "cross_velocity_detrended_m_s"
            ],
            burst[
                "along_velocity_detrended_m_s"
            ],
            fs,
        )

        if f is None:
            continue

        # --------------------------------------------
        # Match depth-averaged tidal current
        # --------------------------------------------
        pos = current["time"].searchsorted(
            burst_midtime
        )

        candidates = []

        for j in (pos - 1, pos):

            if 0 <= j < len(current):

                dt = abs(
                    current.iloc[j]["time"]
                    - burst_midtime
                )

                candidates.append(
                    (dt, j)
                )

        if not candidates:
            continue

        dt, j = min(
            candidates,
            key=lambda x: x[0],
        )

        if dt > current_match_tolerance:
            continue

        current_row = current.iloc[j]

        U_cross = current_row[
            "depth_avg_cross_shore_m_s"
        ]

        U_along = current_row[
            "depth_avg_alongshore_m_s"
        ]

        U_speed = current_row[
            "depth_avg_current_speed_m_s"
        ]

        # Direction of depth-averaged tidal current
        #
        # 0 deg  = cross-shore axis
        # 90 deg = alongshore axis
        #
        # Convert VECTOR direction to an AXIS direction
        # so flood/ebb along the same axis become equivalent.
        theta_current = (
            np.degrees(
                np.arctan2(
                    U_along,
                    U_cross,
                )
            )
            % 180.0
        )

        # --------------------------------------------
        # Each frequency band
        # --------------------------------------------
        for band_name, (
            f_low,
            f_high,
        ) in bands.items():

            var_u = integrate_band(
                f,
                Suu,
                f_low,
                f_high,
            )

            var_v = integrate_band(
                f,
                Svv,
                f_low,
                f_high,
            )

            # covariance comes from REAL part of Suv
            cov_uv = integrate_band(
                f,
                np.real(Suv),
                f_low,
                f_high,
            )

            (
                theta_axis,
                A_ellipse,
                lambda_max,
                lambda_min,
            ) = principal_axis_from_covariance(
                var_u,
                var_v,
                cov_uv,
            )

            # ----------------------------------------
            # Smallest angle between oscillation axis
            # and tidal-current axis: 0-90 degrees
            # ----------------------------------------
            delta_theta = abs(
                theta_axis
                - theta_current
            )

            delta_theta = min(
                delta_theta,
                180.0 - delta_theta,
            )

            axis_records.append(
                {
                    "case_id": case_id,
                    "cell_number": cell,
                    "height_m": z_cell,
                    "burst_number": burst_number,
                    "time": burst_midtime,

                    "band": band_name,

                    "f_low": f_low,
                    "f_high": f_high,

                    "var_u": var_u,
                    "var_v": var_v,
                    "cov_uv": cov_uv,

                    "theta_axis_deg": theta_axis,
                    "A_ellipse": A_ellipse,

                    "lambda_max": lambda_max,
                    "lambda_min": lambda_min,

                    "U_DA": U_speed,
                    "U_cross": U_cross,
                    "U_along": U_along,

                    "theta_current_deg": (
                        theta_current
                    ),

                    "delta_theta_deg": (
                        delta_theta
                    ),
                }
            )


axis_df = pd.DataFrame(
    axis_records
)

print(
    "\nPrincipal-axis records:",
    len(axis_df),
)


# ============================================================
# PLOT 1:
# TIME-VARYING PRINCIPAL AXIS + CURRENT AXIS
#
# One figure per frequency band.
# Four rows = four ADCP cells.
# ============================================================

for band_name in bands:

    dband = axis_df.loc[
        axis_df["band"] == band_name
    ].copy()

    if dband.empty:
        continue

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, cell in zip(
        axes,
        cells_to_plot,
    ):

        d = dband.loc[
            dband["cell_number"] == cell
        ].sort_values("time")

        if d.empty:
            continue

        z = d["height_m"].median()

        # Principal oscillation axis
        ax.scatter(
            d["time"],
            d["theta_axis_deg"],
            s=8,
            label="Oscillation principal axis",
        )

        # Tidal-current axis
        ax.plot(
            d["time"],
            d["theta_current_deg"],
            linewidth=1.2,
            alpha=1.0,
            label="Depth-averaged current axis",
        )

        ax.set_ylim(
            0,
            180,
        )

        ax.set_yticks(
            [0, 45, 90, 135, 180]
        )

        ax.set_ylabel(
            "Axis angle\n(deg)"
        )

        ax.set_title(
            f"Cell {cell} "
            f"(z={z:.3f} m)"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[0].legend(
        loc="best",
        fontsize=9,
    )

    axes[-1].set_xlabel(
        "Time"
    )

    f_low, f_high = bands[
        band_name
    ]

    fig.suptitle(
        f"Frame {frame_number}: "
        f"{band_name} principal oscillation axis "
        f"({f_low:.3f}-{f_high:.3f} Hz) "
        "and tidal-current axis",
        fontsize=14,
    )

    plt.show()


# ============================================================
# PLOT 2:
# A_ELLIPSE THROUGH TIME
#
# Useful quality / interpretability check:
#
# A_ellipse near 1:
#     strongly directional oscillation,
#     principal axis is meaningful.
#
# A_ellipse near 0:
#     nearly isotropic horizontal motion,
#     principal-axis direction is poorly defined.
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 9),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, band_name in zip(
    axes,
    bands.keys(),
):

    for cell in cells_to_plot:

        d = axis_df.loc[
            (axis_df["band"] == band_name)
            & (
                axis_df["cell_number"]
                == cell
            )
        ].sort_values("time")

        if d.empty:
            continue

        z = d["height_m"].median()

        ax.plot(
            d["time"],
            d["A_ellipse"],
            linewidth=1,
            label=(
                f"Cell {cell} "
                f"(z={z:.3f} m)"
            ),
        )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        r"$A_{\mathrm{ellipse}}$"
    )

    ax.set_title(
        band_name
    )

    ax.grid(
        True,
        alpha=0.25,
    )

axes[0].legend(
    loc="best",
    fontsize=9,
)

axes[-1].set_xlabel(
    "Time"
)

fig.suptitle(
    f"Frame {frame_number}: principal-axis anisotropy\n"
    r"$A_{\mathrm{ellipse}}="
    r"\frac{\lambda_{\max}-\lambda_{\min}}"
    r"{\lambda_{\max}+\lambda_{\min}}$",
    fontsize=14,
)

plt.show()

# ============================================================
# FILTER PRINCIPAL-AXIS RESULTS BY ANISOTROPY
#
# Keep only bursts with:
#     A_ellipse >= 0.4
#
# This removes bursts where the horizontal velocity field
# is too isotropic for the principal-axis direction to be
# interpreted confidently.
# ============================================================

Aellipse_min = 0.5

axis_valid = axis_df.loc[
    axis_df["A_ellipse"] >= Aellipse_min
].copy()

print(
    f"\nKeeping principal-axis results with "
    f"A_ellipse >= {Aellipse_min:.2f}"
)

print(
    f"Valid records: {len(axis_valid)} "
    f"out of {len(axis_df)}"
)


# ============================================================
# PLOT 1:
# TIME-VARYING OSCILLATION AXIS + CURRENT AXIS
# ONLY WHERE A_ellipse >= 0.4
#
# One figure per band
# Four rows = four cells
# ============================================================

for band_name in bands:

    dband = axis_valid.loc[
        axis_valid["band"] == band_name
    ].copy()

    if dband.empty:
        continue

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, cell in zip(
        axes,
        cells_to_plot,
    ):

        d = dband.loc[
            dband["cell_number"] == cell
        ].sort_values("time")

        if d.empty:
            ax.set_title(
                f"Cell {cell}: no bursts with "
                f"A_ellipse >= {Aellipse_min}"
            )
            continue

        z = d["height_m"].median()

        # Oscillation principal axis
        ax.scatter(
            d["time"],
            d["theta_axis_deg"],
            s=14,
            label="Oscillation principal axis",
        )

        # Corresponding tidal-current axis
        ax.scatter(
            d["time"],
            d["theta_current_deg"],
            s=14,
            color="black",
            label="Depth-averaged current axis",
        )

        ax.set_ylim(
            0,
            180,
        )

        ax.set_yticks(
            [0, 45, 90, 135, 180]
        )

        ax.set_ylabel(
            "Axis angle\n(deg)"
        )

        ax.set_title(
            f"Cell {cell} "
            f"(z={z:.3f} m), "
            f"N={len(d)}"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[0].legend(
        loc="best",
        fontsize=9,
    )

    axes[-1].set_xlabel(
        "Time"
    )

    f_low, f_high = bands[
        band_name
    ]

    fig.suptitle(
        f"Frame {frame_number}: "
        f"{band_name} principal axis "
        f"({f_low:.3f}-{f_high:.3f} Hz), "
        rf"$A_{{ellipse}}\geq{Aellipse_min:.1f}$",
        fontsize=14,
    )

    plt.show()

# ============================================================
# OPTIONAL SUMMARY STATISTICS
# ============================================================

summary = (
    axis_valid
    .groupby(
        [
            "band",
            "cell_number",
        ]
    )
    .agg(
        N=("delta_theta_deg", "size"),
        median_delta_theta_deg=(
            "delta_theta_deg",
            "median",
        ),
        mean_delta_theta_deg=(
            "delta_theta_deg",
            "mean",
        ),
        median_Aellipse=(
            "A_ellipse",
            "median",
        ),
        median_U_DA=(
            "U_DA",
            "median",
        ),
    )
    .reset_index()
)

print(
    "\nPrincipal-axis summary "
    f"for A_ellipse >= {Aellipse_min:.2f}:"
)

print(
    summary.to_string(
        index=False
    )
)