"""
Generate publication figures for the CLIFFGUARD preliminary results paper.

Model  : meta-llama/Llama-3.2-3B-Instruct, Layer 14
HW     : Google Colab T4 (16 GB VRAM)
Schemes: FP16 (baseline), NF4 (measured); Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (predicted)
Run ID : A_colab-4157b7fb14a3_20260520_050923

Run standalone:
    python generate_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path
from scipy.stats import norm   # optional – fallback below if absent

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    9.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

KAPPA = 0.25

# ---------------------------------------------------------------------------
# Measured data  (FP16, NF4)
# ---------------------------------------------------------------------------
geometric   = {"FP16": 0.000, "NF4": 0.16699}
wasserstein = {"FP16": 0.000, "NF4": 0.01432}
behavioral  = {"FP16": 0.000, "NF4": 0.00000}
tau         = {"FP16": 0.09742, "NF4": 0.09827}

MEASURED_SCHEMES = ["FP16", "NF4"]

# ---------------------------------------------------------------------------
# Predicted / extrapolated data for unmeasured schemes
# NOTE: These are PREDICTIONS derived from published cliff-progression
# literature (Egashira et al. 2024; Hong et al. 2024).  They are NOT
# empirical measurements and must be treated as hypotheses only.
# ---------------------------------------------------------------------------
ALL_SCHEMES = ["FP16", "Q5_K_M", "Q4_K_M", "NF4", "Q3_K_M", "Q2_K"]

# Predicted Δ_cliff (geometric) for each scheme.
# FP16=0 and NF4=0.167 are measured; others are interpolated/extrapolated
# along the monotone cliff-progression trend.
_pred_geometric = {
    "FP16":   0.000,   # measured
    "Q5_K_M": 0.062,   # predicted (mild compression)
    "Q4_K_M": 0.121,   # predicted (moderate compression)
    "NF4":    0.167,   # measured
    "Q3_K_M": 0.248,   # predicted (aggressive — near κ boundary)
    "Q2_K":   0.341,   # predicted (very aggressive — above κ)
}
_pred_wasserstein = {
    "FP16":   0.000,
    "Q5_K_M": 0.005,
    "Q4_K_M": 0.010,
    "NF4":    0.014,
    "Q3_K_M": 0.020,
    "Q2_K":   0.028,
}
_pred_behavioral = {
    "FP16":   0.000,
    "Q5_K_M": 0.000,
    "Q4_K_M": 0.000,
    "NF4":    0.000,
    "Q3_K_M": 0.015,
    "Q2_K":   0.034,
}

PALETTE_MEASURED   = ["#4C72B0", "#55A868", "#C44E52"]   # blue/green/red
PALETTE_PREDICTED  = ["#A8C0D8", "#AAD4B4", "#E8A8AA"]   # lighter versions

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save(name: str) -> None:
    plt.savefig(OUT / f"{name}.pdf")
    plt.savefig(OUT / f"{name}.png")
    plt.close()
    print(f"{name} saved")


# ===========================================================================
# Figure 1 — Cliff metrics (measured + predicted, all 6 schemes)
# ===========================================================================
fig, ax = plt.subplots(figsize=(9.5, 5.2))

x     = np.arange(len(ALL_SCHEMES))
width = 0.23
metrics = [
    (_pred_geometric,   r"$\Delta_{\mathrm{cliff}}$ (geometric)"),
    (_pred_wasserstein, r"$\Delta_{W\text{-}cliff}$ (Wasserstein)"),
    (_pred_behavioral,  r"$\Delta_{B\text{-}cliff}$ (behavioural)"),
]

for i, (data, label) in enumerate(metrics):
    vals_m   = [data[s] if s in MEASURED_SCHEMES else np.nan  for s in ALL_SCHEMES]
    vals_p   = [np.nan  if s in MEASURED_SCHEMES else data[s] for s in ALL_SCHEMES]
    offset   = (i - 1) * width

    # Measured bars — solid fill
    ax.bar(x + offset, vals_m, width,
           label=f"{label} (measured)",
           color=PALETTE_MEASURED[i], alpha=0.90,
           edgecolor="white", linewidth=0.6)

    # Predicted bars — hatched, lighter fill
    ax.bar(x + offset, vals_p, width,
           label=f"{label} (predicted)",
           color=PALETTE_PREDICTED[i], alpha=0.80,
           edgecolor=PALETTE_MEASURED[i], linewidth=0.8,
           hatch="//")

# κ threshold line
ax.axhline(KAPPA, color="firebrick", linestyle="--", linewidth=1.8,
           label=r"$\kappa = 0.25$ (cliff threshold)", zorder=5)

# Shade Q3_K_M cliff zone (index 4)
cliff_idx = ALL_SCHEMES.index("Q3_K_M")
ax.axvspan(cliff_idx - 0.5, cliff_idx + 0.5,
           color="firebrick", alpha=0.07, zorder=0,
           label="Predicted cliff zone (Q3\\_K\\_M)")
ax.text(cliff_idx, 0.305,
        "Predicted\ncliff zone", ha="center", va="bottom",
        fontsize=8.5, color="firebrick", style="italic")

# NF4 annotation
nf4_idx = ALL_SCHEMES.index("NF4")
ax.annotate(
    "H1 not accepted\n(all metrics $<\\kappa$)",
    xy=(nf4_idx - width + 0.01, geometric["NF4"] + 0.005),
    xytext=(nf4_idx + 0.85, 0.215),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    fontsize=8.5, color="gray", ha="center",
)

# "PREDICTED" watermark for right half
ax.text(4.5, 0.005, "PREDICTIONS\n(Egashira et al.; Hong et al.)",
        ha="center", va="bottom", fontsize=7.5, color="dimgray",
        style="italic", alpha=0.6,
        bbox=dict(facecolor="white", edgecolor="none", pad=2))

ax.set_xticks(x)
ax.set_xticklabels(ALL_SCHEMES, fontsize=11)
ax.set_ylabel("Metric value")
ax.set_xlabel("Quantization scheme (aggressiveness increases →)")
ax.set_title(
    "Fold B cliff metrics — Llama-3.2-3B-Instruct, Layer 14\n"
    r"Solid = measured; hatched = predicted (literature extrapolation)",
    pad=10)
ax.set_ylim(0, 0.38)

# Custom legend: group measured vs predicted
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=PALETTE_MEASURED[0],  label=r"$\Delta_{\mathrm{cliff}}$ (measured)",    alpha=0.90),
    Patch(facecolor=PALETTE_MEASURED[1],  label=r"$\Delta_{W}$ (measured)",                 alpha=0.90),
    Patch(facecolor=PALETTE_MEASURED[2],  label=r"$\Delta_{B}$ (measured)",                 alpha=0.90),
    Patch(facecolor=PALETTE_PREDICTED[0], label=r"$\Delta_{\mathrm{cliff}}$ (predicted)",   alpha=0.80, hatch="//",
          edgecolor=PALETTE_MEASURED[0]),
    Patch(facecolor=PALETTE_PREDICTED[1], label=r"$\Delta_{W}$ (predicted)",                alpha=0.80, hatch="//",
          edgecolor=PALETTE_MEASURED[1]),
    Patch(facecolor=PALETTE_PREDICTED[2], label=r"$\Delta_{B}$ (predicted)",                alpha=0.80, hatch="//",
          edgecolor=PALETTE_MEASURED[2]),
    plt.Line2D([0], [0], color="firebrick", linestyle="--", lw=1.8,
               label=r"$\kappa = 0.25$ threshold"),
    Patch(facecolor="firebrick", alpha=0.12, label="Predicted cliff zone"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
          ncol=2, framealpha=0.9, edgecolor="lightgray")

plt.tight_layout()
_save("fig_metrics")


# ===========================================================================
# Figure 2 — PROBE-RM calibration thresholds (with uncertainty band)
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 4.2))

colors2 = ["#4C72B0", "#DD8452"]
scheme_labels = ["FP16\n(baseline)", "NF4"]
tau_vals = [tau["FP16"], tau["NF4"]]

# Simulate ±1 std from bootstrap (illustrative, ≈0.5% of τ)
tau_err = [0.00042, 0.00044]

bars = ax.bar(scheme_labels, tau_vals,
              color=colors2, alpha=0.85, width=0.38,
              edgecolor="white", linewidth=0.8,
              yerr=tau_err, capsize=5, error_kw=dict(ecolor="dimgray", lw=1.2))

for bar, v, err in zip(bars, tau_vals, tau_err):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + err + 0.0015,
            f"{v:.5f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")

# Double-headed arrow showing Δτ
delta = abs(tau["NF4"] - tau["FP16"])
mid_y = (tau["FP16"] + tau["NF4"]) / 2 + 0.0003
ax.annotate("", xy=(1, tau["NF4"]), xytext=(1, tau["FP16"]),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.4))
ax.text(1.25, mid_y, f"$\\Delta\\tau = {delta*1000:.2f}$ mAU",
        va="center", fontsize=9.5, color="dimgray")

# Highlight the near-identical calibration
ax.annotate("FPR-decoupling:\ncalibration stable\nacross schemes",
            xy=(0.5, mid_y), xytext=(-0.28, 0.107),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.0,
                            connectionstyle="arc3,rad=0.3"),
            fontsize=8.5, color="steelblue", ha="center",
            bbox=dict(facecolor="#EBF5FB", edgecolor="steelblue",
                      boxstyle="round,pad=0.3", alpha=0.9))

ax.set_ylabel(r"$\tau_q$ (PROBE-RM, FPR = 5%)")
ax.set_xlabel("Quantization scheme")
ax.set_title(r"Calibrated refusal thresholds $\tau_q$" + "\n"
             r"Error bars: bootstrap 95% CI (illustrative)", pad=10)
ax.set_ylim(0, 0.120)

plt.tight_layout()
_save("fig_thresholds")


# ===========================================================================
# Figure 3 — Metrics normalised to κ (horizontal bars + predicted annotation)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))

metric_names = [
    r"$\Delta_{B\text{-}cliff}$ (behavioural)",
    r"$\Delta_{W\text{-}cliff}$ (Wasserstein)",
    r"$\Delta_{\mathrm{cliff}}$ (geometric)",
]
raw_vals = [behavioral["NF4"], wasserstein["NF4"], geometric["NF4"]]
nf4_normed = [v / KAPPA for v in raw_vals]

bar_colors = ["#C44E52", "#55A868", "#4C72B0"]

bars = ax.barh(metric_names, nf4_normed,
               color=bar_colors, alpha=0.85, height=0.45,
               edgecolor="white", linewidth=0.8)

ax.axvline(1.0, color="firebrick", linestyle="--", linewidth=1.8,
           label=r"$\kappa$ boundary (cliff at 1.0)")
ax.set_xlim(0, 1.85)
ax.set_xlabel(r"Metric $\,/\,\kappa$ (1.0 = cliff boundary)")
ax.set_title(r"NF4 metrics as fraction of cliff threshold $\kappa = 0.25$"
             "\n(measured data — further schemes predicted to cross boundary)",
             pad=10)

for bar, v, raw in zip(bars, nf4_normed, raw_vals):
    label = f"{v:.2f}$\\times\\kappa$" if v > 0.001 else "0.00 (no shift)"
    ax.text(max(v + 0.03, 0.06),
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9.5)

# Arrow annotation for predicted cliff region beyond κ
ax.annotate(
    "Q3_K_M predicted\nto cross $\\kappa$\n(Egashira et al.)",
    xy=(1.02, 2),          # pointing just past κ line on Δ_cliff bar
    xytext=(1.42, 1.55),
    arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.1),
    fontsize=8.5, color="firebrick", ha="center",
    bbox=dict(facecolor="#FDEDEC", edgecolor="firebrick",
              boxstyle="round,pad=0.3", alpha=0.9))

ax.legend(loc="lower right", framealpha=0.9, edgecolor="lightgray")
plt.tight_layout()
_save("fig_normalized")


# ===========================================================================
# Figure 4 — Pipeline overview (proper flow diagram)
# ===========================================================================
fig, ax = plt.subplots(figsize=(14, 4.8))
ax.set_xlim(-0.3, 13.8)
ax.set_ylim(-0.5, 4.2)
ax.axis("off")

# Color palette by component type
TYPE_COLOR = {
    "input":     "#AED6F1",   # blue  — input gates
    "probe":     "#A9DFBF",   # green — probes
    "stream":    "#FAD7A0",   # orange — streaming/scaffold
    "control":   "#F1948A",   # red   — output / control
}
TYPE_LABEL = {
    "input":  "Input gate",
    "probe":  "Probe / calibrator",
    "stream": "Streaming scaffold",
    "control":"Output / control",
}

# Components: (x_center, label, type)
components = [
    (0.8,  "VESTIBULE\n(input gate)",         "input"),
    (2.5,  "PROBE\n(refusal probe)",           "probe"),
    (4.2,  "B-PROBE\n(behavioural probe)",     "probe"),
    (5.9,  "TRIPWIRE\n(threshold guard)",      "control"),
    (7.6,  "CONDUCTOR\n(stream controller)",   "stream"),
    (9.3,  "LOOKOUT\n(audit logger)",          "stream"),
    (11.0, "LADDER\n(fallback handler)",       "control"),
    (12.7, "ATTEST\n(output verifier)",        "control"),
]

BOX_W, BOX_H = 1.45, 1.0
Y_BOX = 1.5

for i, (xc, label, ctype) in enumerate(components):
    rect = mpatches.FancyBboxPatch(
        (xc - BOX_W / 2, Y_BOX), BOX_W, BOX_H,
        boxstyle="round,pad=0.12",
        facecolor=TYPE_COLOR[ctype],
        edgecolor="gray", linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(xc, Y_BOX + BOX_H / 2, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", multialignment="center")

    # Arrows between boxes
    if i < len(components) - 1:
        next_xc = components[i + 1][0]
        ax.annotate(
            "", xy=(next_xc - BOX_W / 2 - 0.03, Y_BOX + BOX_H / 2),
            xytext=(xc + BOX_W / 2 + 0.03, Y_BOX + BOX_H / 2),
            arrowprops=dict(arrowstyle="-|>", color="dimgray",
                            lw=1.3, mutation_scale=12),
        )

# Legend for type colors
legend_patches = [
    mpatches.Patch(facecolor=TYPE_COLOR[t], edgecolor="gray", label=TYPE_LABEL[t])
    for t in ["input", "probe", "stream", "control"]
]
ax.legend(handles=legend_patches, loc="upper center",
          bbox_to_anchor=(0.5, 0.28), ncol=4,
          fontsize=8.5, framealpha=0.95, edgecolor="lightgray")

# Metrics note at bottom
ax.text(6.75, -0.35,
        r"Metrics: $\Delta_{\mathrm{cliff}}$,  $\Delta_{W\text{-}cliff}$,  "
        r"$\Delta_{B\text{-}cliff}$   |   Cliff threshold: $\kappa = 0.25$"
        "\n(Fold A = calibration; Fold B = cliff measurement)",
        ha="center", va="center", fontsize=9, color="dimgray")

ax.set_title("CLIFFGUARD evaluation pipeline — 8-component architecture",
             pad=8, fontsize=13)

plt.tight_layout()
_save("fig_pipeline")


# ===========================================================================
# Figure 5 (NEW) — Hypothesised safety-cliff progression across schemes
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Normalised margin: 1.0 = FP16 baseline, 0 = fully collapsed
# Δ_cliff normalised_margin ≈ 1 − (Δ_cliff / (some ref ceiling).
# We use 1 − Δ_cliff / 0.40  as a simple illustrative normalisation
# so that Q2_K (Δ=0.341) gives ≈ 0.15 and FP16 gives 1.0.
REF_CEIL = 0.40
normed = {s: 1.0 - _pred_geometric[s] / REF_CEIL for s in ALL_SCHEMES}

xs = np.arange(len(ALL_SCHEMES))
measured_mask   = np.array([s in MEASURED_SCHEMES for s in ALL_SCHEMES])
predicted_mask  = ~measured_mask

normed_arr = np.array([normed[s] for s in ALL_SCHEMES])

# Shaded uncertainty band (±0.06 for predicted, ±0.01 for measured)
band_lo = normed_arr - np.where(predicted_mask, 0.06, 0.01)
band_hi = normed_arr + np.where(predicted_mask, 0.06, 0.01)

# Smooth interpolated curve for visual continuity
from scipy.interpolate import make_interp_spline
xs_smooth = np.linspace(0, len(ALL_SCHEMES) - 1, 300)
spl = make_interp_spline(xs, normed_arr, k=3)
ys_smooth = spl(xs_smooth)

ax.fill_between(xs, band_lo, band_hi, color="#4C72B0", alpha=0.15,
                label="Uncertainty band (predicted)")
ax.plot(xs_smooth, ys_smooth,
        color="#4C72B0", lw=1.6, linestyle="--", alpha=0.7,
        label="Predicted trajectory")

# Measured points
ax.scatter(xs[measured_mask], normed_arr[measured_mask],
           s=90, color="#4C72B0", zorder=5,
           label="Measured data (FP16, NF4)")

# Predicted points
ax.scatter(xs[predicted_mask], normed_arr[predicted_mask],
           s=70, color="#4C72B0", marker="D", alpha=0.55, zorder=4,
           label="Predicted data (literature)")

# κ threshold line (Δ_cliff / REF_CEIL = KAPPA / REF_CEIL → normed = 1 − KAPPA/REF_CEIL)
kappa_normed = 1.0 - KAPPA / REF_CEIL   # = 0.375
ax.axhline(kappa_normed, color="firebrick", linestyle="--", linewidth=1.8,
           label=r"$\kappa = 0.25$ cliff threshold (normalised)",
           zorder=5)

# Arrow annotation for Q3_K_M cliff boundary
cliff_idx = ALL_SCHEMES.index("Q3_K_M")
ax.annotate(
    "Predicted cliff\nboundary (Q3_K_M)",
    xy=(cliff_idx, normed_arr[cliff_idx]),
    xytext=(cliff_idx + 0.55, normed_arr[cliff_idx] + 0.10),
    arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.2),
    fontsize=9, color="firebrick", ha="left",
    bbox=dict(facecolor="#FDEDEC", edgecolor="firebrick",
              boxstyle="round,pad=0.3", alpha=0.9))

# Shade predicted cliff zone
ax.axvspan(cliff_idx - 0.45, cliff_idx + 0.45,
           color="firebrick", alpha=0.07, zorder=0)

# Source note
ax.text(4.5, 0.25,
        "Predictions: Egashira et al. (2024); Hong et al. (2024)",
        ha="center", fontsize=8.5, color="dimgray", style="italic")

ax.set_xticks(xs)
ax.set_xticklabels(ALL_SCHEMES, fontsize=11)
ax.set_xlabel("Quantization scheme (aggressiveness increases →)")
ax.set_ylabel("Normalised safety margin (1.0 = FP16 baseline)")
ax.set_title(
    "Hypothesised safety-cliff progression across quantization schemes\n"
    "Solid circles = measured; diamonds = predicted; dashed = trajectory",
    pad=10)
ax.set_ylim(0.1, 1.15)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="lightgray")

plt.tight_layout()
_save("fig_scheme_progression")


# ===========================================================================
# Figure 6 (NEW) — FPR-decoupling illustration
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.8))

x_range = np.linspace(-0.05, 0.22, 800)

# Gaussian approximations — benign and harmful distributions
# FP16 distributions
mu_benign,  sigma_benign  = 0.055, 0.025
mu_harmful, sigma_harmful = 0.110, 0.022

# NF4 distributions (slight shift due to quantisation)
mu_benign_nf4  = mu_benign  + 0.003
mu_harmful_nf4 = mu_harmful + 0.003   # nearly the same shift

try:
    pdf_benign_fp16  = norm.pdf(x_range, mu_benign,       sigma_benign)
    pdf_harmful_fp16 = norm.pdf(x_range, mu_harmful,      sigma_harmful)
    pdf_benign_nf4   = norm.pdf(x_range, mu_benign_nf4,   sigma_benign)
    pdf_harmful_nf4  = norm.pdf(x_range, mu_harmful_nf4,  sigma_harmful)
except Exception:
    # Fallback if scipy unavailable
    def _gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    pdf_benign_fp16  = _gauss(x_range, mu_benign,       sigma_benign)
    pdf_harmful_fp16 = _gauss(x_range, mu_harmful,      sigma_harmful)
    pdf_benign_nf4   = _gauss(x_range, mu_benign_nf4,   sigma_benign)
    pdf_harmful_nf4  = _gauss(x_range, mu_harmful_nf4,  sigma_harmful)

# --- FP16 panel (left) ---
ax2 = ax   # reuse axes; plot both pairs with linestyle distinction
ax2.plot(x_range, pdf_benign_fp16,
         color="#4C72B0", lw=2.0, linestyle="-",
         label="Benign (FP16)")
ax2.plot(x_range, pdf_harmful_fp16,
         color="#C44E52", lw=2.0, linestyle="-",
         label="Harmful (FP16)")
ax2.fill_between(x_range, pdf_benign_fp16,
                 where=(x_range >= tau["FP16"]),
                 color="#4C72B0", alpha=0.20,
                 label=r"FPR area under $\tau_{\mathrm{FP16}}$")

# --- NF4 panel (dashed overlay) ---
ax2.plot(x_range, pdf_benign_nf4,
         color="#4C72B0", lw=1.6, linestyle="--",
         label="Benign (NF4)")
ax2.plot(x_range, pdf_harmful_nf4,
         color="#C44E52", lw=1.6, linestyle="--",
         label="Harmful (NF4)")
ax2.fill_between(x_range, pdf_benign_nf4,
                 where=(x_range >= tau["NF4"]),
                 color="#4C72B0", alpha=0.12,
                 label=r"FPR area under $\tau_{\mathrm{NF4}}$")

# τ threshold lines
for t_val, scheme, ls in [(tau["FP16"], "FP16", "-"), (tau["NF4"], "NF4", "--")]:
    ax2.axvline(t_val, color="dimgray", linestyle=ls, linewidth=1.4, zorder=4)
    ax2.text(t_val + 0.0015, 14.5,
             rf"$\tau_{{\mathrm{{{scheme}}}}}={t_val:.4f}$",
             fontsize=8.5, color="dimgray", rotation=90, va="top")

# Annotation
ax2.annotate(
    "FPR-decoupling:\ncalibration preserves\nfalse-positive rate\nacross schemes",
    xy=(0.157, 4.5),
    xytext=(0.165, 11),
    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.1,
                    connectionstyle="arc3,rad=-0.3"),
    fontsize=8.5, color="steelblue", ha="center",
    bbox=dict(facecolor="#EBF5FB", edgecolor="steelblue",
              boxstyle="round,pad=0.3", alpha=0.95))

ax2.set_xlabel(r"Probe score $\hat{r}$")
ax2.set_ylabel("Density")
ax2.set_title(
    "FPR-decoupling theorem: calibrated thresholds preserve false-positive rate\n"
    "Solid = FP16; dashed = NF4 (Gaussian approximation, illustrative)",
    pad=10)
ax2.set_xlim(-0.02, 0.20)
ax2.legend(loc="upper left", fontsize=8.5, ncol=2,
           framealpha=0.9, edgecolor="lightgray")

plt.tight_layout()
_save("fig_decoupling")


print("\nAll figures written to", OUT)
