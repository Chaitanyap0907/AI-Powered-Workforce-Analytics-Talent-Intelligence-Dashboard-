"""
Phase 10: SageMaker Training Entrypoint
------------------------------------------
This script is written to the SageMaker Script Mode contract, so it runs
EITHER:
  (a) locally, for testing:
        python aws/sagemaker_train.py --train ./data/processed --model-dir ./models_sm
  (b) inside a managed SageMaker Training Job, where SageMaker automatically
      sets SM_CHANNEL_TRAIN and SM_MODEL_DIR and calls this same script.

It trains the same 3 candidates as attrition_model.py (Logistic Regression,
Random Forest, XGBoost), logs metrics in the format SageMaker Experiments /
CloudWatch can capture, and saves the winning model + metadata to model-dir
so it can be registered in SageMaker Model Registry and deployed to an
endpoint (see launch_sagemaker_training.py and deploy_endpoint.py).
"""

import argparse
import os
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

BASE_FEATURES = [
    "TenureYears", "Age", "SatisfactionScore", "EngagementScore",
    "WorkLifeBalanceScore", "CurrentEmployeeRating", "PerformanceScoreNumeric",
    "EducationLevelNumeric", "Recruitment_YearsofExperience", "Recruitment_DesiredSalary",
    "Training_DurationDays", "Training_Cost",
]


def get_feature_columns(df: pd.DataFrame):
    dummy_prefixes = (
        "BusinessUnit_", "EmployeeType_", "PayZone_", "EmployeeClassificationType_",
        "DepartmentType_", "Division_", "Gender_", "Race_", "MaritalStatus_",
        "JobFunctionDescription_",
    )
    dummy_cols = [c for c in df.columns if c.startswith(dummy_prefixes)]
    return BASE_FEATURES + dummy_cols


def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker injects these as env vars automatically inside a Training Job;
    # defaults let the exact same script run locally for testing.
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/processed"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./models_sm"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1": round(f1_score(y_test, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
    }


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.train) / "workforce_master.csv"
    df = pd.read_csv(data_path)
    features = get_feature_columns(df)
    X = df[features].fillna(0)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, models = [], {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate(lr, X_test_scaled, y_test, "LogisticRegression"))
    models["LogisticRegression"] = (lr, True)

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "RandomForest"))
    models["RandomForest"] = (rf, False)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate(xgb, X_test, y_test, "XGBoost"))
    models["XGBoost"] = (xgb, False)

    # SageMaker Training Jobs parse metrics from stdout via regex when you
    # configure metric_definitions on the Estimator - print in a simple,
    # greppable format for that purpose.
    for r in results:
        print(f"[METRIC] model={r['Model']} accuracy={r['Accuracy']} "
              f"precision={r['Precision']} recall={r['Recall']} "
              f"f1={r['F1']} roc_auc={r['ROC-AUC']}")

    best = max(results, key=lambda r: r["ROC-AUC"])
    best_model, needs_scaling = models[best["Model"]]
    print(f"[BEST] {best['Model']} selected with ROC-AUC={best['ROC-AUC']}")

    joblib.dump(best_model, model_dir / "model.pkl")
    joblib.dump(features, model_dir / "model_features.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(needs_scaling, model_dir / "needs_scaling.pkl")

    metadata = {"best_model": best["Model"], "metrics": results}
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model artifacts saved to {model_dir}")


if __name__ == "__main__":
    main()
