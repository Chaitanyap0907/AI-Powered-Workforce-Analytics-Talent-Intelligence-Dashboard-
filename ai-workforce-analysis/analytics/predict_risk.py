"""
Phase 9: Predictive Workforce Risk Engine
--------------------------------------------
Loads the trained model, scores every CURRENTLY ACTIVE employee for
attrition probability, assigns a risk category, computes SHAP-based
"top reasons," and generates a plain-English retention recommendation.
Writes data/processed/employee_predictions.csv, consumed by app.py.
"""

import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "workforce_master.csv"
MODELS_DIR = BASE_DIR / "models"
OUT_PATH = BASE_DIR / "data" / "processed" / "employee_predictions.csv"

# Human-readable labels for the raw feature names used in SHAP explanations
FEATURE_LABELS = {
    "TenureYears": "short tenure",
    "Age": "age",
    "SatisfactionScore": "low satisfaction score",
    "EngagementScore": "low engagement",
    "WorkLifeBalanceScore": "poor work-life balance",
    "CurrentEmployeeRating": "low performance rating",
    "PerformanceScoreNumeric": "low performance score",
    "EducationLevelNumeric": "education level",
    "Recruitment_YearsofExperience": "experience level",
    "Recruitment_DesiredSalary": "desired salary at hire",
    "Training_DurationDays": "training investment",
    "Training_Cost": "training investment",
}


def label_for(feature_name):
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    for prefix in ["DepartmentType_", "BusinessUnit_", "EmployeeType_", "Division_",
                   "JobFunctionDescription_", "Gender_", "Race_", "MaritalStatus_",
                   "PayZone_", "EmployeeClassificationType_"]:
        if feature_name.startswith(prefix):
            return f"{prefix[:-1].replace('Type','')} = {feature_name[len(prefix):]}"
    return feature_name


def risk_category(prob):
    if prob >= 0.6:
        return "High Risk"
    elif prob >= 0.3:
        return "Medium Risk"
    return "Low Risk"


def recommendation_for(top_reasons, risk_cat):
    if risk_cat == "Low Risk":
        return "Maintain current engagement practices; no immediate action needed."

    reason_text = " and ".join(top_reasons[:2])
    actions = []
    joined = " ".join(top_reasons).lower()
    if "satisfaction" in joined or "engagement" in joined:
        actions.append("schedule a 1:1 to discuss engagement and career goals")
    if "tenure" in joined:
        actions.append("provide onboarding/mentorship support given short tenure")
    if "work-life" in joined:
        actions.append("review workload and flexible-work options")
    if "performance" in joined:
        actions.append("offer coaching or a targeted development plan")
    if "salary" in joined:
        actions.append("review compensation against market benchmarks")
    if not actions:
        actions.append("check in on overall engagement and role fit")

    urgency = "Prioritize immediate action." if risk_cat == "High Risk" else "Monitor and follow up."
    return f"Key drivers: {reason_text}. Recommended: {'; '.join(actions)}. {urgency}"


def run_predictions():
    df = pd.read_csv(DATA_PATH)
    features = joblib.load(MODELS_DIR / "model_features.pkl")
    model = joblib.load(MODELS_DIR / "best_attrition_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    needs_scaling = joblib.load(MODELS_DIR / "needs_scaling.pkl")
    best_model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")

    # Score ALL employees (dashboard can filter to active); this mirrors
    # the pattern of scoring the full current workforce for planning purposes.
    active_df = df[df["EmployeeStatus"].isin(["Active", "Leave of Absence", "Future Start"])].copy()
    X = active_df[features].fillna(0)
    X_model = scaler.transform(X) if needs_scaling else X

    active_df["AttritionProbability"] = model.predict_proba(X_model)[:, 1]
    active_df["RiskCategory"] = active_df["AttritionProbability"].apply(risk_category)

    # SHAP explainability (tree explainer works for RF/XGBoost; for LR fall back to coefficients)
    top_reasons_list = []
    try:
        if best_model_name in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                sv = shap_values[1]  # class 1 (attrition) contributions
            elif shap_values.ndim == 3:
                sv = shap_values[:, :, 1]  # (n_samples, n_features, n_classes) -> class 1
            else:
                sv = shap_values
            dummy_cols = set(c for c in features if any(
                c.startswith(p) for p in ["DepartmentType_", "BusinessUnit_", "EmployeeType_",
                                           "Division_", "JobFunctionDescription_", "Gender_",
                                           "Race_", "MaritalStatus_", "PayZone_",
                                           "EmployeeClassificationType_"]
            ))
            for i in range(len(X)):
                row_shap = pd.Series(sv[i], index=features)
                row_x = X.iloc[i]
                # For one-hot dummy columns, only a meaningful reason if TRUE (value==1) for this employee
                valid = row_shap[(row_shap > 0) & (~row_shap.index.isin(dummy_cols) | (row_x == 1))]
                top3 = valid.sort_values(ascending=False).head(3)
                top_reasons_list.append([label_for(f) for f in top3.index])
        else:
            coefs = pd.Series(model.coef_[0], index=features)
            dummy_cols = set(c for c in features if any(
                c.startswith(p) for p in ["DepartmentType_", "BusinessUnit_", "EmployeeType_",
                                           "Division_", "JobFunctionDescription_", "Gender_",
                                           "Race_", "MaritalStatus_", "PayZone_",
                                           "EmployeeClassificationType_"]
            ))
            for i in range(len(X)):
                row_x = X.iloc[i]
                contrib = coefs * row_x
                valid = contrib[(contrib > 0) & (~contrib.index.isin(dummy_cols) | (row_x == 1))]
                top3 = valid.sort_values(ascending=False).head(3)
                top_reasons_list.append([label_for(f) for f in top3.index])
    except Exception as e:
        print(f"SHAP explanation failed, falling back to generic reasons: {e}")
        top_reasons_list = [["multiple factors"] for _ in range(len(X))]

    active_df["TopReasons"] = ["; ".join(r) if r else "no strong single driver" for r in top_reasons_list]
    active_df["Recommendation"] = [
        recommendation_for(r, cat) for r, cat in zip(top_reasons_list, active_df["RiskCategory"])
    ]

    output_cols = [
        "EmployeeID", "FirstName", "LastName", "Title", "DepartmentType", "BusinessUnit",
        "EmployeeType", "Gender", "TenureYears", "Age", "SatisfactionScore", "EngagementScore",
        "WorkLifeBalanceScore", "PerformanceScore", "CurrentEmployeeRating",
        "AttritionProbability", "RiskCategory", "TopReasons", "Recommendation",
    ]
    result = active_df[output_cols].sort_values("AttritionProbability", ascending=False)
    result.to_csv(OUT_PATH, index=False)

    counts = result["RiskCategory"].value_counts()
    total = len(result)
    print(f"Total Employees Scored: {total}")
    for cat in ["High Risk", "Medium Risk", "Low Risk"]:
        n = counts.get(cat, 0)
        print(f"{cat}: {n} employees ({n/total:.1%})")
    print(f"\nSaved -> {OUT_PATH}")
    return result


if __name__ == "__main__":
    run_predictions()
