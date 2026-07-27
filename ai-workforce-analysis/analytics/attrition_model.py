"""
Phase 6-7: Feature Engineering & ML Model Development
-------------------------------------------------------
Trains Logistic Regression, Random Forest, and XGBoost models to predict
employee attrition, evaluates each on Accuracy/Precision/Recall/F1/ROC-AUC,
selects the best performer, and saves it + its feature list + a SHAP
explainer background sample for use by predict_risk.py and the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "workforce_master.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Core predictive features: engagement/satisfaction survey data, tenure,
# performance, recruitment profile, and department/type dummies.
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


def load_data():
    df = pd.read_csv(DATA_PATH)
    features = get_feature_columns(df)
    X = df[features].fillna(0)
    y = df["Attrition"]
    return X, y, features


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    return metrics


def train_and_select():
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    models = {}

    # 1. Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate(lr, X_test_scaled, y_test, "Logistic Regression"))
    models["Logistic Regression"] = (lr, True)  # needs scaling

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "Random Forest"))
    models["Random Forest"] = (rf, False)

    # 3. XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate(xgb, X_test, y_test, "XGBoost"))
    models["XGBoost"] = (xgb, False)

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\nMODEL COMPARISON\n" + "=" * 60)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model, needs_scaling = models[best_name]
    print(f"\nBEST PERFORMING MODEL SELECTED: {best_name}")

    joblib.dump(best_model, MODELS_DIR / "best_attrition_model.pkl")
    joblib.dump(features, MODELS_DIR / "model_features.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(needs_scaling, MODELS_DIR / "needs_scaling.pkl")
    joblib.dump(best_name, MODELS_DIR / "best_model_name.pkl")
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    # Save a small background sample for SHAP (unscaled - tree models used for SHAP)
    X_train.sample(min(200, len(X_train)), random_state=42).to_csv(
        MODELS_DIR / "shap_background.csv", index=False
    )

    print(f"\nSaved: models/best_attrition_model.pkl, model_features.pkl, scaler.pkl")
    return best_model, features, results_df


if __name__ == "__main__":
    train_and_select()
