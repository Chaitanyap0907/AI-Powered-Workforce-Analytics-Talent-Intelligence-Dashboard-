"""
Phase 10: Unified Risk Prediction Client
------------------------------------------
Used by predict_risk.py and app.py. If the SAGEMAKER_ENDPOINT_NAME
environment variable is set, sends requests to the live SageMaker
endpoint (deploy_endpoint.py). Otherwise, transparently falls back to
the local best_attrition_model.pkl so the whole project still runs
with zero AWS setup.
"""

import os
import json
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def using_sagemaker_endpoint() -> bool:
    return bool(os.environ.get("SAGEMAKER_ENDPOINT_NAME"))


def _predict_via_sagemaker(row: dict) -> dict:
    import boto3  # imported lazily so boto3 isn't required for local-only use

    runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    response = runtime.invoke_endpoint(
        EndpointName=os.environ["SAGEMAKER_ENDPOINT_NAME"],
        ContentType="application/json",
        Body=json.dumps(row),
    )
    return json.loads(response["Body"].read())


class LocalModelClient:
    """Loads the local .pkl artifacts once and scores rows exactly like predict_risk.py."""

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "best_attrition_model.pkl")
        self.features = joblib.load(MODELS_DIR / "model_features.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.needs_scaling = joblib.load(MODELS_DIR / "needs_scaling.pkl")

    def predict_row(self, feature_row: pd.Series) -> dict:
        X = pd.DataFrame([feature_row[self.features].fillna(0)], columns=self.features)
        X_model = self.scaler.transform(X) if self.needs_scaling else X
        probability = float(self.model.predict_proba(X_model)[0, 1])
        risk = "High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low"
        return {"risk": risk, "probability": round(probability, 4)}


_local_client = None


def predict_attrition_risk(feature_row: pd.Series) -> dict:
    """
    Single entry point used by the rest of the app. Returns
    {"risk": "High"/"Medium"/"Low", "probability": float}
    exactly matching the SageMaker endpoint contract from the spec,
    regardless of which backend actually served it.
    """
    if using_sagemaker_endpoint():
        payload = {
            "satisfaction": feature_row.get("SatisfactionScore", 0),
            "engagement": feature_row.get("EngagementScore", 0),
            "tenure": feature_row.get("TenureYears", 0),
            "department": feature_row.get("DepartmentType", ""),
        }
        return _predict_via_sagemaker(payload)

    global _local_client
    if _local_client is None:
        _local_client = LocalModelClient()
    return _local_client.predict_row(feature_row)


if __name__ == "__main__":
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "workforce_master.csv")
    sample = df.iloc[0]
    print(f"Using {'SageMaker endpoint' if using_sagemaker_endpoint() else 'local model'} backend")
    print(predict_attrition_risk(sample))
