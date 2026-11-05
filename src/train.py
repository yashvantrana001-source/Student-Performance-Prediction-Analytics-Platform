"""Train, compare and save the best model.

Run:  python src/train.py
Outputs:
  models/best_model.pkl     -> sklearn Pipeline (Imputer + Scaler + Classifier)
  models/metrics.json       -> metrics used by the Streamlit app
  reports/model_results.csv -> comparison table
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "student_performance.csv"
FEATURES = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score",
]
TARGET = "result"


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def main():
    df = pd.read_csv(DATA).drop_duplicates()
    X = df[FEATURES]
    y = (df[TARGET] == "Pass").astype(int)  # 1 = Pass, 0 = Fail

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    results, fitted = [], {}
    for name, model in candidates.items():
        pipe = make_pipeline(model).fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        fitted[name] = pipe
        results.append({"model": name, "test_accuracy": round(acc, 4)})
        print(f"{name:28s} accuracy = {acc:.3f}")

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best = fitted[best_name]
    y_pred = best.predict(X_test)

    # Feature importance: coefficients for linear model, importances for trees
    est = best.named_steps["model"]
    if hasattr(est, "coef_"):
        importance = dict(zip(FEATURES, est.coef_[0].round(4).tolist()))
    else:
        importance = dict(zip(FEATURES, est.feature_importances_.round(4).tolist()))

    metrics = {
        "best_model": best_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Fail", "Pass"], output_dict=True
        ),
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "best_model.pkl")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    results_df.to_csv(ROOT / "reports" / "model_results.csv", index=False)
    print(f"\nBest model: {best_name}  ->  models/best_model.pkl")


if __name__ == "__main__":
    main()
