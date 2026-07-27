"""
Phase 11: Amazon Bedrock AI HR Assistant
------------------------------------------
Implements the three features from the spec:
  1. explain_employee_risk(employee_id)   - "Why is employee 1024 high risk?"
  2. answer_workforce_question(question)  - "Which department needs attention?"
  3. generate_retention_strategy()        - retention action plan from risk counts

Dual-mode backend:
  - If USE_BEDROCK=1 and AWS credentials are configured, calls Claude via
    Amazon Bedrock (bedrock-runtime.invoke_model).
  - Otherwise, calls the Anthropic API directly (same model, same prompts) -
    this is what lets the assistant work today, with only ANTHROPIC_API_KEY
    set, before you've wired up Bedrock at all.

Either backend only ever receives AGGREGATE statistics + the single
employee's own record when explicitly asked about that employee - never
the full employee table - to limit PII exposure.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_NumpyEncoder)

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "data" / "processed" / "employee_predictions.csv"
MASTER_PATH = BASE_DIR / "data" / "processed" / "workforce_master.csv"

SYSTEM_PROMPT = (
    "You are an HR workforce analytics assistant embedded in a company dashboard. "
    "Answer using ONLY the JSON data context provided - never invent numbers. "
    "Be concise, specific, and actionable. Use plain language a non-technical "
    "HR manager would understand."
)

# Bedrock model ID for Claude (adjust to the version enabled in your account)
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _use_bedrock() -> bool:
    return os.environ.get("USE_BEDROCK", "0") == "1"


def _call_llm(user_message: str, max_tokens: int = 700) -> str:
    if _use_bedrock():
        return _call_bedrock(user_message, max_tokens)
    return _call_anthropic_api(user_message, max_tokens)


def _call_bedrock(user_message: str, max_tokens: int) -> str:
    import boto3  # lazy import - boto3 only required when USE_BEDROCK=1

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }
    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    return "\n".join(b["text"] for b in payload.get("content", []) if b.get("type") == "text")


def _call_anthropic_api(user_message: str, max_tokens: int) -> str:
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM backend configured. Set ANTHROPIC_API_KEY (direct Anthropic API) "
            "or USE_BEDROCK=1 with AWS credentials (Amazon Bedrock)."
        )
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        },
        timeout=30,
    )
    data = resp.json()
    if "content" not in data:
        raise RuntimeError(f"Anthropic API error: {data}")
    return "\n".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


# ---------------------------------------------------------------- Data helpers
def _load_predictions() -> pd.DataFrame:
    return pd.read_csv(PRED_PATH)


def _load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_PATH, parse_dates=["StartDate", "ExitDate", "DateOfBirth"])


def _department_summary(preds: pd.DataFrame) -> dict:
    return (
        preds.groupby("DepartmentType")
        .agg(
            headcount=("EmployeeID", "count"),
            avg_attrition_prob=("AttritionProbability", "mean"),
            high_risk_count=("RiskCategory", lambda s: (s == "High Risk").sum()),
        )
        .round(3)
        .to_dict(orient="index")
    )


# ---------------------------------------------------------------- Feature 1
def explain_employee_risk(employee_id: int) -> str:
    """'Why is employee 1024 high risk?'"""
    preds = _load_predictions()
    row = preds[preds["EmployeeID"] == employee_id]
    if row.empty:
        return f"No prediction found for Employee ID {employee_id}."
    row = row.iloc[0]

    context = {
        "employee_id": int(row["EmployeeID"]),
        "department": row["DepartmentType"],
        "risk_category": row["RiskCategory"],
        "attrition_probability": round(float(row["AttritionProbability"]), 3),
        "satisfaction_score": float(row["SatisfactionScore"]),
        "engagement_score": float(row["EngagementScore"]),
        "work_life_balance_score": float(row["WorkLifeBalanceScore"]),
        "tenure_years": float(row["TenureYears"]),
        "performance_rating": float(row["CurrentEmployeeRating"]),
        "top_model_drivers": row["TopReasons"],
    }
    prompt = (
        f"DATA CONTEXT (single employee, already anonymized to ID only):\n{_dumps(context)}\n\n"
        f"QUESTION: Why is employee {employee_id} at {row['RiskCategory'].lower()}? "
        "Explain in 3-5 bullet points using the specific numbers given."
    )
    return _call_llm(prompt)


# ---------------------------------------------------------------- Feature 2
def answer_workforce_question(question: str) -> str:
    """'Which department needs attention?' / general workforce questions."""
    preds = _load_predictions()
    master = _load_master()
    active = master[master["EmployeeStatus"].isin(["Active", "Leave of Absence", "Future Start"])]

    context = {
        "total_active_employees": int(len(active)),
        "overall_historical_attrition_rate": round(float(master["Attrition"].mean()), 3),
        "risk_category_counts": preds["RiskCategory"].value_counts().to_dict(),
        "department_summary": _department_summary(preds),
        "avg_satisfaction": round(float(active["SatisfactionScore"].mean()), 2),
        "avg_engagement": round(float(active["EngagementScore"].mean()), 2),
        "avg_performance_rating": round(float(active["CurrentEmployeeRating"].mean()), 2),
        "dei_gender": active["Gender"].value_counts().to_dict(),
        "dei_race": active["Race"].value_counts().to_dict(),
        "training_outcomes": active["Training_Outcome"].value_counts().to_dict(),
    }
    prompt = f"DATA CONTEXT:\n{_dumps(context)}\n\nQUESTION: {question}"
    return _call_llm(prompt)


# ---------------------------------------------------------------- Feature 3
def generate_retention_strategy(department: str = None) -> str:
    """Retention Strategy Generator - overall or scoped to one department."""
    preds = _load_predictions()
    scope = preds if department is None else preds[preds["DepartmentType"] == department]

    counts = scope["RiskCategory"].value_counts().to_dict()
    top_reasons = (
        scope["TopReasons"].str.split("; ").explode().value_counts().head(10).to_dict()
    )
    context = {
        "scope": department or "entire workforce",
        "employees_in_scope": int(len(scope)),
        "risk_counts": counts,
        "most_common_risk_drivers": top_reasons,
    }
    prompt = (
        f"DATA CONTEXT:\n{_dumps(context)}\n\n"
        f"TASK: Generate a numbered retention strategy (4-6 concrete recommended actions) "
        f"for {context['scope']}, prioritized by the most common risk drivers shown."
    )
    return _call_llm(prompt)


if __name__ == "__main__":
    preds = _load_predictions()
    sample_id = int(preds.sort_values("AttritionProbability", ascending=False).iloc[0]["EmployeeID"])
    print(f"[TEST] Backend: {'Bedrock' if _use_bedrock() else 'Direct Anthropic API'}")
    print(f"[TEST] explain_employee_risk({sample_id}) ->")
    try:
        print(explain_employee_risk(sample_id))
    except RuntimeError as e:
        print(f"  (skipped - {e})")
