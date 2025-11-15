"""
Quick check of model health metrics with simple alerting.

Keeping this script tiny so I can run it during coffee breaks.
"""

from __future__ import annotations

import json
from pathlib import Path

# TODO: make these configurable via CLI once someone complains loudly enough.
DEFAULT_THRESHOLDS = {
    "roc_auc": 0.8,
    "recall": 0.2,
}


def main(metrics_path: Path, thresholds: dict[str, float]) -> int:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Couldn't find metrics at {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    overall = metrics.get("overall", {})
    alerts = []

    for model, stats in overall.items():
        # Yep, double loop is deliberate—readability beats shaving milliseconds here.
        for metric_name, min_value in thresholds.items():
            val = stats.get(metric_name)
            if val is None:
                continue
            if val < min_value:
                alerts.append(f"{model}: {metric_name}={val:.3f} < {min_value:.3f}")

    if alerts:
        print("ALERT: Model quality below threshold")
        for msg in alerts:
            print(f" - {msg}")
        # TODO: pipe this into Slack/email if it ever becomes mission critical.
        return 1

    print("All models within thresholds.")
    return 0


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    metrics = base_dir / "reports" / "model_metrics.json"
    raise SystemExit(main(metrics, DEFAULT_THRESHOLDS))

