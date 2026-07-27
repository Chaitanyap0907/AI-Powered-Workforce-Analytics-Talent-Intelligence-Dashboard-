"""
Phase 2-4: Data Extraction, Cleaning & Feature Engineering Pipeline
--------------------------------------------------------------------
Loads the three raw HR data sources (employee_master, training_performance,
recruitment), merges them on EmployeeID, cleans and engineers features,
and writes a single ML-ready dataset to data/processed/workforce_master.csv

In production (AWS), these raw CSVs would instead be the result of
Athena queries pulled via boto3 - see aws_setup_guide.md for that code.
Locally, we read directly from data/raw/ so the rest of the pipeline
(EDA, ML, dashboard) is identical either way.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TODAY = pd.Timestamp("2026-07-26")  # snapshot date for tenure/age calcs


def load_raw():
    emp = pd.read_csv(RAW_DIR / "employee_master.csv")
    perf = pd.read_csv(RAW_DIR / "training_performance.csv")
    rec = pd.read_csv(RAW_DIR / "recruitment.csv")
    return emp, perf, rec


def clean_employee_master(emp: pd.DataFrame) -> pd.DataFrame:
    df = emp.copy()

    # Parse dates
    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
    df["ExitDate"] = pd.to_datetime(df["ExitDate"], errors="coerce")
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")

    # Drop duplicate employee records if any
    df = df.drop_duplicates(subset="EmployeeID")

    # Attrition label: voluntary/for-cause termination = left the company
    df["Attrition"] = df["EmployeeStatus"].isin(
        ["Voluntarily Terminated", "Terminated for Cause"]
    ).astype(int)

    # Tenure in years (as of exit date if left, else as of snapshot date)
    end_date = df["ExitDate"].fillna(TODAY)
    df["TenureYears"] = ((end_date - df["StartDate"]).dt.days / 365.25).round(2)
    df["TenureYears"] = df["TenureYears"].clip(lower=0)

    # Age
    df["Age"] = ((TODAY - df["DateOfBirth"]).dt.days / 365.25).round(0)

    # Fill missing termination info for active employees
    df["TerminationType"] = df["TerminationType"].fillna("N/A - Active")
    df["TerminationDescription"] = df["TerminationDescription"].fillna("N/A - Active")

    # Strip whitespace from string cols
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    return df


def clean_training_performance(perf: pd.DataFrame) -> pd.DataFrame:
    df = perf.copy()
    df = df.drop_duplicates(subset="EmployeeID")
    df["SurveyDate"] = pd.to_datetime(df["SurveyDate"], errors="coerce")
    df["Training_Date"] = pd.to_datetime(df["Training_Date"], errors="coerce")

    # Numeric performance rating already present as CurrentEmployeeRating (1-5 scale)
    perf_map = {"PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4}
    df["PerformanceScoreNumeric"] = df["PerformanceScore"].map(perf_map)

    df["Training_DurationDays"] = pd.to_numeric(df["Training_DurationDays"], errors="coerce").fillna(0)
    df["Training_Cost"] = pd.to_numeric(df["Training_Cost"], errors="coerce").fillna(0)

    return df


def clean_recruitment(rec: pd.DataFrame) -> pd.DataFrame:
    df = rec.copy()
    df = df.drop_duplicates(subset="EmployeeID")
    df["Recruitment_ApplicationDate"] = pd.to_datetime(df["Recruitment_ApplicationDate"], errors="coerce")
    df["Recruitment_DateofBirth"] = pd.to_datetime(df["Recruitment_DateofBirth"], errors="coerce")

    edu_map = {"High School": 1, "Bachelor's Degree": 2, "Master's Degree": 3, "PhD": 4}
    df["EducationLevelNumeric"] = df["Recruitment_EducationLevel"].map(edu_map)

    df["Recruitment_YearsofExperience"] = pd.to_numeric(
        df["Recruitment_YearsofExperience"], errors="coerce"
    ).fillna(0)
    df["Recruitment_DesiredSalary"] = pd.to_numeric(
        df["Recruitment_DesiredSalary"], errors="coerce"
    ).fillna(df["Recruitment_DesiredSalary"].median())

    # Keep only the columns useful downstream (drop PII not needed for analytics)
    keep = [
        "EmployeeID", "Recruitment_ApplicationDate", "Recruitment_EducationLevel",
        "EducationLevelNumeric", "Recruitment_YearsofExperience", "Recruitment_DesiredSalary",
        "Recruitment_JobTitle", "Recruitment_Status", "Recruitment_State", "Recruitment_Country",
    ]
    return df[keep]


def build_master_dataset():
    emp, perf, rec = load_raw()

    emp_c = clean_employee_master(emp)
    perf_c = clean_training_performance(perf)
    rec_c = clean_recruitment(rec)

    df = emp_c.merge(perf_c, on="EmployeeID", how="left")
    df = df.merge(rec_c, on="EmployeeID", how="left")

    # Department/type categories already exist as DepartmentType / EmployeeType
    # One-hot encode key categoricals for ML use later (kept alongside raw labels)
    cat_cols = [
        "BusinessUnit", "EmployeeType", "PayZone", "EmployeeClassificationType",
        "DepartmentType", "Division", "Gender", "Race", "MaritalStatus",
        "JobFunctionDescription",
    ]
    encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)

    df_final = pd.concat([df, encoded], axis=1)

    out_path = PROCESSED_DIR / "workforce_master.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved cleaned, merged, feature-engineered dataset -> {out_path}")
    print(f"Shape: {df_final.shape}")
    print(f"Attrition rate: {df_final['Attrition'].mean():.1%}")
    return df_final


if __name__ == "__main__":
    build_master_dataset()
