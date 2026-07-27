"""
Phase 10: SageMaker Inference Handler
------------------------------------------
Implements the SageMaker Scikit-learn container's inference contract
(model_fn / input_fn / predict_fn / output_fn). Package this alongside
model.pkl / model_features.pkl / scaler.pkl / needs_scaling.pkl (all
produced by sagemaker_train.py) and deploy with an SKLearnModel /
SKLearnPredictor as shown in deploy_endpoint.py.

Endpoint contract (matches the JSON in the Phase 10 spec):

  Request:
    {"satisfaction": 2, "tenure": 1.5, "department": "Sales", ...}

  Response:
    {"risk": "High", "probability": 0.87}
"""

import json
import joblib
import pandas as pd
from pathlib import Path

# Map the friendly request field names -> the model's engineered feature names.
FIELD_MAP = {
    "satisfaction": "SatisfactionScore",
    "engagement": "EngagementScore",
    "work_life_balance": "WorkLifeBalanceScore",
    "tenure": "TenureYears",
    "age": "Age",
    "performance_rating": "CurrentEmployeeRating",
    "education_level": "EducationLevelNumeric",
    "years_experience": "Recruitment_YearsofExperience",
    "desired_salary": "Recruitment_DesiredSalary",
    "training_duration_days": "Training_DurationDays",
    "training_cost": "Training_Cost",
}
# department / business_unit / gender / etc. map onto one-hot dummy columns,
# e.g. department="Sales" -> feature "DepartmentType_Sales" = 1
DUMMY_FIELD_PREFIXES = {
    "department": "DepartmentType_",
    "business_unit": "BusinessUnit_",
    "employee_type": "EmployeeType_",
    "division": "Division_",
    "gender": "Gender_",
    "race": "Race_",
    "marital_status": "MaritalStatus_",
    "pay_zone": "PayZone_",
    "job_function": "JobFunctionDescription_",
}


def model_fn(model_dir):
    """Called once when the endpoint container starts."""
    model_dir = Path(model_dir)
    return {
        "model": joblib.load(model_dir / "model.pkl"),
        "features": joblib.load(model_dir / "model_features.pkl"),
        "scaler": joblib.load(model_dir / "scaler.pkl"),
        "needs_scaling": joblib.load(model_dir / "needs_scaling.pkl"),
    }


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


def _build_feature_row(payload: dict, features: list) -> pd.DataFrame:
    row = {f: 0.0 for f in features}
    for key, value in payload.items():
        if key in FIELD_MAP and FIELD_MAP[key] in row:
            row[FIELD_MAP[key]] = value
        elif key in DUMMY_FIELD_PREFIXES:
            col = f"{DUMMY_FIELD_PREFIXES[key]}{value}"
            if col in row:
                row[col] = 1.0
    return pd.DataFrame([row], columns=features)


def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    features = model_artifacts["features"]
    scaler = model_artifacts["scaler"]
    needs_scaling = model_artifacts["needs_scaling"]

    X = _build_feature_row(input_data, features)
    X_model = scaler.transform(X) if needs_scaling else X

    probability = float(model.predict_proba(X_model)[0, 1])
    if probability >= 0.6:
        risk = "High"
    elif probability >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"
    return {"risk": risk, "probability": round(probability, 4)}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept


if __name__ == "__main__":
    # Local smoke test - simulates exactly what the endpoint container does.
    artifacts = model_fn("./models_sm_test")
    sample = {"satisfaction": 2, "tenure": 1.5, "department": "Sales", "engagement": 1}
    parsed = input_fn(json.dumps(sample))
    result = predict_fn(parsed, artifacts)
    body, _ = output_fn(result)
    print(body)
