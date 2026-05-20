"""
Generate publication figures for the CLIFFGUARD preliminary results paper.

Model : meta-llama/Llama-3.2-3B-Instruct, Layer 14
Hardware: Google Colab T4 (16 GB VRAM)
Schemes : FP16 (baseline), NF4
Run ID  : A_colab-4157b7fb14a3_20260520_050923
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KAPPA = 0.25

# ---------------------------------------------------------------------------
# Raw results
# ---------------------------------------------------------------------------
geometric   = {"FP16": 0.000,     "NF4": 0.16699}
wasserstein = {"FP16": 0.000,     "NF4": 0.01432}
behavioral  = {"FP16": 0.000,     "NF4": 0.00000}
tau         = {"FP16": 0.09742,   "NF4": 0.09827}
SCHEMES     = ["FP16", "NF4"]

# ---------------------------------------------------------------------------
# Figure 1 — Cliff metrics grouped bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))

x       = np.arange(len(SCHEMES))
width   = 0.22
palette = ["#4C72B0", "#55A868", "#C44E52"]
metrics = [
    (geometric,   r"$\Delta_{\mathrm{cliff}}$ (geometric)"),
    (wasserstein, r"$\Delta_{W\text{-cliff}}$ (Wasserstein)"),
    (behavioral,  r"$\Delta_{B\text{-cliff}}$ (behavioural)"),
]

for i, (data, label) in enumerate(metrics):
    vals = [data[s] for s in SCHEMES]
    ax.bar(x + (i - 1) * width, vals, width,
           label=label, color=palette[i], alpha=0.85,
           edgecolor="white", linewidth=0.6)

ax.axhline(KAPPA, color="firebrick", linestyle="--", linewidth=1.6,
           label=r"$\kappa = 0.25$ (cliff threshold)")

ax.annotate(
    "H1 not accepted\n(all metrics $<\\kappa$)",
    xy=(1 - width, geometric["NF4"] + 0.005),
    xytext=(1.25, 0.21),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    fontsize=9, color="gray", ha="center",
)

ax.set_xticks(x)
ax.set_xticklabels(SCHEMES, fontsize=12)
ax.set_ylabel("Metric value")
ax.set_xlabel("Quantization scheme")
ax.set_title("Fold B cliff metrics — Llama-3.2-3B-Instruct, Layer 14", pad=10)
ax.legend(loc="upper left")
ax.set_ylim(0, 0.33)

plt.tight_layout()
plt.savefig(OUT / "fig_metrics.pdf")
plt.savefig(OUT / "fig_metrics.png")
plt.close()
print("fig_metrics saved")

# ---------------------------------------------------------------------------
# Figure 2 — PROBE-RM calibration thresholds
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 3.8))

colors2 = ["#4C72B0", "#DD8452"]
bars = ax.bar(SCHEMES, [tau[s] for s in SCHEMES],
              color=colors2, alpha=0.85, width=0.4, edgecolor="white")

for bar, s in zip(bars, SCHEMES):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0008,
            f"{tau[s]:.5f}", ha="center", va="bottom", fontsize=10)

delta = abs(tau["NF4"] - tau["FP16"])
mid_y = (tau["FP16"] + tau["NF4"]) / 2
ax.annotate(
    "",
    xy=(1, tau["NF4"]), xytext=(1, tau["FP16"]),
    arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
)
ax.text(1.22, mid_y, f"$\\Delta\\tau = {delta*1000:.2f}$ mAU",
        va="center", fontsize=9, color="dimgray")

ax.set_ylabel(r"$\tau_q$ (PROBE-RM, FPR = 5\%)")
ax.set_xlabel("Quantization scheme")
ax.set_title(r"Calibrated refusal thresholds $\tau_q$", pad=10)
ax.set_ylim(0, 0.115)

plt.tight_layout()
plt.savefig(OUT / "fig_thresholds.pdf")
plt.savefig(OUT / "fig_thresholds.png")
plt.close()
print("fig_thresholds saved")

# ---------------------------------------------------------------------------
# Figure 3 — Metrics normalised to kappa (horizontal bar)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.6))

metric_names = [
    r"$\Delta_{\mathrm{cliff}}$ (geometric)",
    r"$\Delta_{W\text{-cliff}}$ (Wasserstein)",
    r"$\Delta_{B\text{-cliff}}$ (behavioural)",
]
nf4_normed = [
    geometric["NF4"]   / KAPPA,
    wasserstein["NF4"] / KAPPA,
    behavioral["NF4"]  / KAPPA,
]

bars = ax.barh(metric_names, nf4_normed,
               color="#4C72B0", alpha=0.82, height=0.45)

ax.axvline(1.0, color="firebrick", linestyle="--", linewidth=1.6,
           label=r"$\kappa$ boundary (cliff)")
ax.set_xlim(0, 1.45)
ax.set_xlabel(r"Metric $\,/\,\kappa\;$ (1.0 = cliff boundary)")
ax.set_title("NF4 metrics as fraction of cliff threshold $\\kappa = 0.25$", pad=10)
ax.legend()

for bar, v, raw in zip(bars, nf4_normed,
                       [geometric["NF4"], wasserstein["NF4"], behavioral["NF4"]]):
    label = f"{v:.2f}$\\times\\kappa$" if v > 0 else "0.00 (no shift)"
    ax.text(v + 0.03, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT / "fig_normalized.pdf")
plt.savefig(OUT / "fig_normalized.png")
plt.close()
print("fig_normalized saved")

# ---------------------------------------------------------------------------
# Figure 4 — Pipeline overview (schematic)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    (0.4,  "Fold A\nCalibration",       "#AED6F1"),
    (2.6,  "r̂ direction\n(Arditi DIM)", "#A9DFBF"),
    (4.8,  r"$\tau_q$ threshold" + "\n(PROBE-RM)",  "#A9DFBF"),
    (7.0,  "Fold B\nCliff Measurement",  "#FAD7A0"),
    (9.2,  "H1 verdict\n(accept/reject)", "#F1948A"),
]

for x0, label, color in boxes:
    rect = mpatches.FancyBboxPatch(
        (x0, 0.8), 1.8, 1.4,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="gray", linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(x0 + 0.9, 1.5, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", multialignment="center")

for x0 in [2.2, 4.4, 6.6, 8.8]:
    ax.annotate("", xy=(x0, 1.5), xytext=(x0 - 0.02, 1.5),
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.3))

ax.text(5.0, 0.35,
        r"Metrics: $\Delta_{\mathrm{cliff}}$,  $\Delta_{W\text{-cliff}}$,  "
        r"$\Delta_{B\text{-cliff}}$   |   Threshold: $\kappa = 0.25$",
        ha="center", va="center", fontsize=9, color="dimgray")

ax.set_title("CLIFFGUARD evaluation pipeline (Phase A–B)", pad=8)

plt.tight_layout()
plt.savefig(OUT / "fig_pipeline.pdf")
plt.savefig(OUT / "fig_pipeline.png")
plt.close()
print("fig_pipeline saved")

print("\nAll figures written to", OUT)
