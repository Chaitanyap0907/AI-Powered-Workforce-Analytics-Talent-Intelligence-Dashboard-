"""
Phase 5: Exploratory Data Analysis
-----------------------------------
Generates six visualization reports (PNG) summarizing workforce
distribution, department analysis, satisfaction, performance,
training, and tenure/retention patterns.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "workforce_master.csv"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load():
    return pd.read_csv(DATA_PATH, parse_dates=["StartDate", "ExitDate", "DateOfBirth"])


def workforce_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    active = df[df["EmployeeStatus"] == "Active"]
    active["EmployeeType"].value_counts().plot.pie(
        ax=axes[0], autopct="%1.0f%%", ylabel="", title="Active Workforce by Employee Type"
    )
    status_counts = df["EmployeeStatus"].value_counts()
    sns.barplot(x=status_counts.values, y=status_counts.index, ax=axes[1], palette="viridis")
    axes[1].set_title("Employee Status Distribution")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "1_workforce_overview.png", dpi=120)
    plt.close()


def department_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dept_counts = df["DepartmentType"].value_counts()
    sns.barplot(x=dept_counts.values, y=dept_counts.index, ax=axes[0], palette="mako")
    axes[0].set_title("Headcount by Department")
    axes[0].set_xlabel("Employees")

    dept_attr = df.groupby("DepartmentType")["Attrition"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=dept_attr.values, y=dept_attr.index, ax=axes[1], palette="rocket")
    axes[1].set_title("Attrition Rate by Department (%)")
    axes[1].set_xlabel("Attrition %")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "2_department_analysis.png", dpi=120)
    plt.close()


def satisfaction_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["SatisfactionScore"], bins=5, kde=False, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Satisfaction Score Distribution")

    sat_by_attr = df.groupby("Attrition")["SatisfactionScore"].mean()
    sns.barplot(x=["Stayed", "Left"], y=sat_by_attr.values, ax=axes[1], palette="coolwarm")
    axes[1].set_title("Avg Satisfaction: Stayed vs Left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "3_satisfaction_analysis.png", dpi=120)
    plt.close()


def performance_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    perf_counts = df["PerformanceScore"].value_counts()
    sns.barplot(x=perf_counts.index, y=perf_counts.values, ax=axes[0], palette="crest")
    axes[0].set_title("Performance Score Distribution")
    axes[0].tick_params(axis="x", rotation=20)

    perf_attr = df.groupby("PerformanceScore")["Attrition"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=perf_attr.index, y=perf_attr.values, ax=axes[1], palette="flare")
    axes[1].set_title("Attrition Rate by Performance Score (%)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "4_performance_analysis.png", dpi=120)
    plt.close()


def training_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    out_counts = df["Training_Outcome"].value_counts()
    sns.barplot(x=out_counts.index, y=out_counts.values, ax=axes[0], palette="Set2")
    axes[0].set_title("Training Outcome Distribution")

    type_cost = df.groupby("Training_Type")["Training_Cost"].mean()
    sns.barplot(x=type_cost.index, y=type_cost.values, ax=axes[1], palette="Set3")
    axes[1].set_title("Avg Training Cost by Type ($)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "5_training_analysis.png", dpi=120)
    plt.close()


def tenure_retention_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["TenureYears"], bins=20, ax=axes[0], color="#55A868")
    axes[0].set_title("Tenure Distribution (Years)")

    tenure_by_attr = df.groupby("Attrition")["TenureYears"].mean()
    sns.barplot(x=["Stayed", "Left"], y=tenure_by_attr.values, ax=axes[1], palette="magma")
    axes[1].set_title("Avg Tenure: Stayed vs Left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "6_tenure_retention_analysis.png", dpi=120)
    plt.close()


def run_eda():
    df = load()
    workforce_overview(df)
    department_analysis(df)
    satisfaction_analysis(df)
    performance_analysis(df)
    training_analysis(df)
    tenure_retention_analysis(df)
    print(f"EDA complete. 6 visualizations saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    run_eda()
