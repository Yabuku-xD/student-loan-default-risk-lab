"""
Generate a concise markdown summary of model watchlists and thresholds.

Usage:
    python src/tools/watchlist_report.py
"""

# I promise to keep this a single-file script until it becomes impossible.
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"
OUTPUT_PATH = BASE_DIR / "reports" / "watchlist_summary.md"
# TODO: support alternate output paths when we automate weekly briefs.


def _format_percentage(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    # Round gently so the markdown table stays readable.
    return f"{value:.2%}"


def _format_float(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    thresholds = metrics.get("thresholds", {})
    watchlists = metrics.get("watchlists", {})

    lines: list[str] = ["# Watchlist Summary", ""]
    # Personal preference: headings mirror model names so stakeholders recognize them.

    for model_name, info in watchlists.items():
        watchlist_rel = info.get("watchlist")
        if not watchlist_rel:
            continue
        watchlist_path = BASE_DIR / watchlist_rel
        if not watchlist_path.exists():
            continue

        df = pd.read_csv(watchlist_path)
        flagged_count = int(len(df))
        avg_default = df["default_rate_pct"].mean()
        avg_net_price = df["net_price"].mean()
        avg_completion = df["completion_rate"].mean()
        threshold_info = thresholds.get(model_name, {})

        lines.append(f"## {model_name.replace('_', ' ').title()}")
        lines.append(
            f"- Recommended threshold: `{threshold_info.get('recommended_threshold', 'n/a')}` "
            f"(Precision {_format_percentage(threshold_info.get('recommended_precision'))}, "
            f"Recall {_format_percentage(threshold_info.get('recommended_recall'))})"
        )
        lines.append(
            f"- Institutions flagged: **{flagged_count}** | "
            f"Avg default {_format_float(avg_default)} | "
            f"Avg net price {_format_currency(avg_net_price)} | "
            f"Avg completion {_format_percentage(avg_completion)}"
        )

        preview_cols = [
            "INSTNM",
            "control_label",
            "region_label",
            "dominant_program",
            "default_rate_pct",
            "risk_score",
        ]
        preview = df[[c for c in preview_cols if c in df.columns]].head(10).copy()
        preview["default_rate_pct"] = preview["default_rate_pct"].round(3)
        if "risk_score" in preview.columns:
            preview["risk_score"] = preview["risk_score"].round(3)

        lines.append("")
        lines.append("Top flagged institutions:")
        lines.append("")
        lines.append(preview.to_markdown(index=False))
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Watchlist summary saved to {OUTPUT_PATH}")
    # TODO: maybe return the markdown string if another tool wants to reuse it.


if __name__ == "__main__":
    main()

