"""Generate a synthetic student performance dataset (5,000 records).

Run:  python src/generate_data.py
Replace data/raw/student_performance.csv with a real dataset anytime;
just keep the same column names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "student_performance.csv"


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(75, 14, n), 30, 100)
    study_hours = np.clip(rng.normal(4.0, 1.8, n), 0, 10)
    previous = np.clip(rng.normal(62, 15, n) + 0.15 * (attendance - 75), 0, 100)
    internal = np.clip(0.5 * previous + 3 * study_hours + rng.normal(22, 10, n), 0, 100)
    assignment = np.clip(0.4 * previous + 2.5 * study_hours + rng.normal(28, 11, n), 0, 100)

    # Hidden "true" relationship + noise -> Pass / Fail
    score = (
        0.30 * internal
        + 0.20 * previous
        + 0.15 * assignment
        + 0.15 * attendance
        + 2.0 * study_hours
        + rng.normal(0, 6, n)
    )
    threshold = np.percentile(score, 30)  # ~30% Fail, ~70% Pass
    result = np.where(score >= threshold, "Pass", "Fail")

    return pd.DataFrame(
        {
            "student_id": [f"S{i:05d}" for i in range(1, n + 1)],
            "attendance": attendance.round(1),
            "internal_marks": internal.round(1),
            "study_hours": study_hours.round(1),
            "previous_performance": previous.round(1),
            "assignment_score": assignment.round(1),
            "result": result,
        }
    )


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows -> {OUT}")
    print(df["result"].value_counts())
