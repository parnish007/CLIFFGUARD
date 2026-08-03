"""Emit LaTeX table bodies from docs/paper/data.json.

Tables are generated for the same reason figures are: a number typed into a
manuscript is a number that can silently disagree with the run that produced it.
Each fragment is \\input{} by main.tex.

Usage:
  python scripts/build_paper_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(value: float | None, decimals: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{decimals}f}"


def bits_label(bits: float) -> str:
    return "FP16" if bits >= 16 else f"{bits:.1f}"


def table_artifact(data: dict[str, Any]) -> str:
    """Judged vs phrase-list harmful compliance, both models side by side."""
    models = [m for m in ("Qwen2.5-3B", "Phi-3.5-mini") if m in data["behavioural"]]
    ref = [r for r in data["behavioural"][models[0]]["rows"] if r["scheme"] != "FP16"]
    ref.sort(key=lambda r: -r["bits"])

    lines = [
        r"\begin{tabular}{l" + "rrc" * len(models) + "}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in models) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                for i in range(len(models))),
        "bits/param & " + " & ".join(
            r"judge & markers & degen." for _ in models) + r" \\",
        r"\midrule",
    ]
    for row in ref:
        cells = []
        for model in models:
            match = next(r for r in data["behavioural"][model]["rows"]
                         if r["scheme"] == row["scheme"])
            flag = r"\ddag" if match["degenerate"] > 0.5 else ""
            cells += [pct(match["unsafe_judge"]), pct(match.get("unsafe_marker")), flag]
        lines.append(f"{bits_label(row['bits'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_capability(data: dict[str, Any]) -> str:
    """GSM8K accuracy with exact intervals and p-values."""
    models = list(data["sector"])
    ref = sorted(data["sector"][models[0]]["rows"], key=lambda r: -r["bits"])
    lines = [
        r"\begin{tabular}{l" + "rl" * len(models) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))),
        "bits/param & " + " & ".join(r"acc.\ (\%) & 95\% CI" for _ in models) + r" \\",
        r"\midrule",
    ]
    for row in ref:
        cells = []
        for model in models:
            match = next((r for r in data["sector"][model]["rows"]
                          if abs(r["bits"] - row["bits"]) < 1e-6), None)
            if match is None:
                cells += ["--", "--"]
                continue
            star = ""
            if match["p_vs_fp16"] is not None and match["p_vs_fp16"] < 0.05:
                star = r"$^{*}$"
            cells += [f"{100 * match['accuracy']:.1f}{star}",
                      f"[{100 * match['ci_low']:.1f}, {100 * match['ci_high']:.1f}]"]
        lines.append(f"{bits_label(row['bits'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_markers(data: dict[str, Any]) -> str:
    """The same completions under four marker lists."""
    model = next((m for m in ("Qwen2.5-3B", "Phi-3.5-mini")
                  if data["behavioural"].get(m, {}).get("marker_variants")), None)
    if model is None:
        return "% no marker-variant data available\n"
    variants = data["behavioural"][model]["marker_variants"]
    schemes = sorted({s for v in variants.values() for s in v},
                     key=lambda s: -int(s.split("_")[1][:-1]))
    lines = [
        r"\begin{tabular}{l" + "r" * len(schemes) + "}",
        r"\toprule",
        "marker list & " + " & ".join(s.split("_")[1][:-1] for s in schemes) + r" \\",
        r"\midrule",
    ]
    for name, row in variants.items():
        safe = name.replace("_", r"\_")
        lines.append(f"{safe} & " + " & ".join(
            pct(row.get(s, 0.0)) for s in schemes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_probe(data: dict[str, Any]) -> str:
    """Frozen-probe retention against judged harmful compliance."""
    models = [m for m in data["transfer"] if m in data["behavioural"]]
    ref = sorted(data["transfer"][models[0]]["rows"], key=lambda r: -r["bits"])
    lines = [
        r"\begin{tabular}{l" + "rr" * len(models) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))),
        "bits/param & " + " & ".join(
            r"$d'$ kept & harmful" for _ in models) + r" \\",
        r"\midrule",
    ]
    for row in ref:
        cells = []
        for model in models:
            t = next((r for r in data["transfer"][model]["rows"]
                      if abs(r["bits"] - row["bits"]) < 1e-6), None)
            b = next((r for r in data["behavioural"][model]["rows"]
                      if abs(r["bits"] - row["bits"]) < 1e-6), None)
            cells += [pct(t["retained"], 0) if t else "--",
                      pct(b["unsafe_judge"]) if b else "--"]
        lines.append(f"{bits_label(row['bits'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/tables"))
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("tab_artifact", table_artifact),
        ("tab_capability", table_capability),
        ("tab_markers", table_markers),
        ("tab_probe", table_probe),
    ):
        (args.out / f"{name}.tex").write_text(builder(data) + "\n", encoding="utf-8")
        print(f"  {name}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
