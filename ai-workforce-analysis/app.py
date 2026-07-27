"""
Phase 8: AI-Powered Workforce Analytics & Talent Intelligence Dashboard
--------------------------------------------------------------------------
Run with:  streamlit run app.py

Requires data/processed/workforce_master.csv and data/processed/employee_predictions.csv
to already exist (run: python analytics/data_pipeline.py, attrition_model.py, predict_risk.py
- or just `python run_pipeline.py` to do all three).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = BASE_DIR / "data" / "processed" / "workforce_master.csv"
PRED_PATH = BASE_DIR / "data" / "processed" / "employee_predictions.csv"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="AI Workforce Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_hosted_secrets() -> None:
    """Expose Streamlit-hosted secrets to the optional AWS/LLM integrations.

    Locally, configuration continues to come from environment variables. On
    Streamlit Community Cloud, secrets are available through ``st.secrets``;
    copying only the known integration keys into the process environment lets
    the existing backend modules use the same configuration in both places.
    """
    try:
        for key in (
            "ANTHROPIC_API_KEY",
            "USE_BEDROCK",
            "AWS_REGION",
            "BEDROCK_MODEL_ID",
            "SAGEMAKER_ENDPOINT_NAME",
        ):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except FileNotFoundError:
        # No local secrets.toml is expected for the offline/demo version.
        pass


load_hosted_secrets()

# ---------------------------------------------------------------- DATA LOAD
@st.cache_data
def load_data():
    master = pd.read_csv(MASTER_PATH, parse_dates=["StartDate", "ExitDate", "DateOfBirth"])
    preds = pd.read_csv(PRED_PATH)
    return master, preds


@st.cache_resource
def load_model_artifacts():
    comparison = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    best_name = joblib.load(MODELS_DIR / "best_model_name.pkl")
    model = joblib.load(MODELS_DIR / "best_attrition_model.pkl")
    features = joblib.load(MODELS_DIR / "model_features.pkl")
    return comparison, best_name, model, features


if not MASTER_PATH.exists() or not PRED_PATH.exists():
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "```\npython analytics/data_pipeline.py\n"
        "python analytics/attrition_model.py\n"
        "python analytics/predict_risk.py\n```"
    )
    st.stop()

master, preds = load_data()
comparison, best_model_name, model, model_features = load_model_artifacts()

active_mask = master["EmployeeStatus"].isin(["Active", "Leave of Absence", "Future Start"])
active = master[active_mask]

st.title("📊 AI Workforce Intelligence Platform")
st.caption("Using AWS SageMaker, Amazon Bedrock & Predictive Analytics for workforce visibility and retention insights")

with st.sidebar.expander("⚙️ Backend status", expanded=False):
    sm_status = "🟢 Live SageMaker endpoint" if os.environ.get("SAGEMAKER_ENDPOINT_NAME") else "⚪ Local model (.pkl)"
    ai_status = "🟢 Amazon Bedrock" if os.environ.get("USE_BEDROCK") == "1" else (
        "🟡 Direct Anthropic API" if os.environ.get("ANTHROPIC_API_KEY") else "⚪ Not configured"
    )
    st.write(f"**Prediction backend:** {sm_status}")
    st.write(f"**AI Assistant backend:** {ai_status}")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive KPIs",
        "Predictive Attrition Risk",
        "Explainable AI (SHAP)",
        "Department Risk Analysis",
        "Employee Analytics",
        "Skills & Career Development",
        "DEI Metrics",
        "Employee Risk Directory",
        "AI Workforce Assistant",
        "HR Policy Assistant (RAG)",
    ],
)

# ==================================================================== PAGE 1
if page == "Executive KPIs":
    st.header("Executive KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Workforce (Active)", f"{len(active):,}")
    high_risk_n = (preds["RiskCategory"] == "High Risk").sum()
    c2.metric("High-Risk Employees", f"{high_risk_n:,}", f"{high_risk_n/len(preds):.1%} of workforce")
    attrition_pct = master["Attrition"].mean()
    c3.metric("Historical Attrition Rate", f"{attrition_pct:.1%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg Satisfaction", f"{active['SatisfactionScore'].mean():.2f} / 5")
    c5.metric("Avg Performance Rating", f"{active['CurrentEmployeeRating'].mean():.2f} / 5")
    c6.metric("Avg Tenure", f"{active['TenureYears'].mean():.1f} yrs")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        status_counts = master["EmployeeStatus"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.bar(status_counts, x="Count", y="Status", orientation="h", title="Employee Status Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        type_counts = active["EmployeeType"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, names="Type", values="Count", title="Active Workforce by Employee Type")
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================== PAGE 2
elif page == "Predictive Attrition Risk":
    st.header("Predictive Attrition Risk")
    c1, c2, c3 = st.columns(3)
    for col, cat, color in zip([c1, c2, c3], ["High Risk", "Medium Risk", "Low Risk"], ["🔴", "🟠", "🟢"]):
        n = (preds["RiskCategory"] == cat).sum()
        col.metric(f"{color} {cat}", f"{n:,}", f"{n/len(preds):.1%}")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = preds["RiskCategory"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"])
        fig = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            title="Risk Category Distribution", labels={"x": "Risk Category", "y": "Employees"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(
            preds, x="AttritionProbability", nbins=30,
            title="Attrition Probability Distribution", color_discrete_sequence=["#4C72B0"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Highest-Risk Employees")
    top15 = preds.sort_values("AttritionProbability", ascending=False).head(15)
    st.dataframe(
        top15[["EmployeeID", "FirstName", "LastName", "DepartmentType", "AttritionProbability",
               "RiskCategory", "TopReasons"]],
        use_container_width=True, hide_index=True,
    )

# ==================================================================== PAGE 3
elif page == "Explainable AI (SHAP)":
    st.header("Explainable AI — Why the Model Predicts What It Predicts")
    st.subheader("Model Comparison")
    st.dataframe(comparison.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]),
                 use_container_width=True)
    st.info(f"**Selected model:** {best_model_name} (best ROC-AUC)")

    st.subheader("Global Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=model_features)
        top20 = importances.sort_values(ascending=False).head(20)
        fig = px.bar(x=top20.values, y=top20.index, orientation="h",
                     title="Top 20 Features Driving Attrition Predictions",
                     labels={"x": "Importance", "y": "Feature"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Feature importances unavailable for this model type.")

    st.subheader("Most Common Risk Drivers Across the Workforce")
    all_reasons = preds["TopReasons"].str.split("; ").explode()
    reason_counts = all_reasons.value_counts().head(15).reset_index()
    reason_counts.columns = ["Reason", "Employees Affected"]
    fig = px.bar(reason_counts, x="Employees Affected", y="Reason", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================== PAGE 4
elif page == "Department Risk Analysis":
    st.header("Department Risk Analysis")
    dept_risk = preds.copy()
    dept_summary = dept_risk.groupby("DepartmentType").agg(
        Headcount=("EmployeeID", "count"),
        AvgAttritionProb=("AttritionProbability", "mean"),
        HighRisk=("RiskCategory", lambda s: (s == "High Risk").sum()),
    ).reset_index().sort_values("AvgAttritionProb", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(dept_summary, x="AvgAttritionProb", y="DepartmentType", orientation="h",
                     title="Avg Attrition Probability by Department",
                     labels={"AvgAttritionProb": "Avg Attrition Probability", "DepartmentType": "Department"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(dept_summary, x="HighRisk", y="DepartmentType", orientation="h",
                     title="High-Risk Headcount by Department",
                     labels={"HighRisk": "High-Risk Employees", "DepartmentType": "Department"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Distribution Heatmap (Department x Risk Category)")
    pivot = dept_risk.groupby(["DepartmentType", "RiskCategory"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                     title="Employee Count by Department & Risk Category")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================== PAGE 5
elif page == "Employee Analytics":
    st.header("Employee Analytics")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(active, x="SatisfactionScore", nbins=5, title="Satisfaction Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(active, x="EngagementScore", nbins=5, title="Engagement Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        perf_counts = active["PerformanceScore"].value_counts().reset_index()
        perf_counts.columns = ["Performance", "Count"]
        fig = px.bar(perf_counts, x="Performance", y="Count", title="Performance Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(active, x="TenureYears", nbins=25, title="Tenure Distribution (Years)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction vs Tenure vs Attrition Risk")
    scatter_df = preds.copy()
    fig = px.scatter(
        scatter_df, x="TenureYears", y="SatisfactionScore", color="RiskCategory",
        color_discrete_map={"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
        hover_data=["EmployeeID", "DepartmentType"], title="Tenure vs Satisfaction, colored by Risk",
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================== PAGE 6
elif page == "Skills & Career Development":
    st.header("Skills, Learning & Career Development")
    col1, col2 = st.columns(2)
    with col1:
        outcome_counts = active["Training_Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig = px.pie(outcome_counts, names="Outcome", values="Count", title="Training Outcome Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        type_cost = active.groupby("Training_Type")["Training_Cost"].mean().reset_index()
        fig = px.bar(type_cost, x="Training_Type", y="Training_Cost", title="Avg Training Cost by Type")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Training Programs")
    prog_counts = active["Training_ProgramName"].value_counts().head(10).reset_index()
    prog_counts.columns = ["Program", "Participants"]
    fig = px.bar(prog_counts, x="Participants", y="Program", orientation="h", title="Most Attended Programs")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Education Level at Hire vs Current Performance")
    edu_perf = active.groupby("Recruitment_EducationLevel")["CurrentEmployeeRating"].mean().reset_index()
    fig = px.bar(edu_perf, x="Recruitment_EducationLevel", y="CurrentEmployeeRating",
                 title="Avg Current Rating by Education Level at Hire")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Failed / Incomplete Training — Career Development Gaps")
    gaps = active[active["Training_Outcome"].isin(["Failed", "Incomplete"])]
    st.dataframe(
        gaps[["EmployeeID", "FirstName", "LastName", "DepartmentType", "Training_ProgramName", "Training_Outcome"]].head(20),
        use_container_width=True, hide_index=True,
    )

# ==================================================================== PAGE 7
elif page == "DEI Metrics":
    st.header("Diversity, Equity & Inclusion Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_counts = active["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(gender_counts, names="Gender", values="Count", title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        race_counts = active["Race"].value_counts().reset_index()
        race_counts.columns = ["Race", "Count"]
        fig = px.pie(race_counts, names="Race", values="Count", title="Race/Ethnicity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        marital_counts = active["MaritalStatus"].value_counts().reset_index()
        marital_counts.columns = ["Status", "Count"]
        fig = px.pie(marital_counts, names="Status", values="Count", title="Marital Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attrition Rate by Gender & Race")
    col4, col5 = st.columns(2)
    with col4:
        g = master.groupby("Gender")["Attrition"].mean().reset_index()
        fig = px.bar(g, x="Gender", y="Attrition", title="Attrition Rate by Gender")
        st.plotly_chart(fig, use_container_width=True)
    with col5:
        r = master.groupby("Race")["Attrition"].mean().reset_index()
        fig = px.bar(r, x="Race", y="Attrition", title="Attrition Rate by Race")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Representation by Department")
    dept_gender = active.groupby(["DepartmentType", "Gender"]).size().reset_index(name="Count")
    fig = px.bar(dept_gender, x="DepartmentType", y="Count", color="Gender", barmode="stack",
                 title="Gender Composition by Department")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pay Zone Distribution by Gender")
    payzone_gender = active.groupby(["PayZone", "Gender"]).size().reset_index(name="Count")
    fig = px.bar(payzone_gender, x="PayZone", y="Count", color="Gender", barmode="group",
                 title="Pay Zone by Gender")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================== PAGE 8
elif page == "Employee Risk Directory":
    st.header("Employee Risk Directory")
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search by name or Employee ID")
    with col2:
        dept_filter = st.multiselect("Filter by Department", sorted(preds["DepartmentType"].unique()))
    with col3:
        risk_filter = st.multiselect("Filter by Risk Category", ["High Risk", "Medium Risk", "Low Risk"])

    filtered = preds.copy()
    if search:
        s = search.lower()
        filtered = filtered[
            filtered["FirstName"].str.lower().str.contains(s)
            | filtered["LastName"].str.lower().str.contains(s)
            | filtered["EmployeeID"].astype(str).str.contains(s)
        ]
    if dept_filter:
        filtered = filtered[filtered["DepartmentType"].isin(dept_filter)]
    if risk_filter:
        filtered = filtered[filtered["RiskCategory"].isin(risk_filter)]

    st.write(f"Showing {len(filtered):,} of {len(preds):,} employees")
    st.dataframe(
        filtered[["EmployeeID", "FirstName", "LastName", "Title", "DepartmentType",
                  "AttritionProbability", "RiskCategory", "TopReasons", "Recommendation"]],
        use_container_width=True, hide_index=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export to CSV", csv, "employee_risk_export.csv", "text/csv")

# ==================================================================== PAGE 9
elif page == "AI Workforce Assistant":
    from analytics.ai_hr_assistant import (
        explain_employee_risk, answer_workforce_question, generate_retention_strategy, _use_bedrock,
    )

    st.header("🤖 AI Workforce Assistant")
    backend_label = "Amazon Bedrock" if _use_bedrock() else "Direct Anthropic API"
    st.caption(f"Phase 11 · Backend: **{backend_label}** · switch with the `USE_BEDROCK` environment variable")

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")) or _use_bedrock()
    if not has_key:
        st.warning(
            "Set `ANTHROPIC_API_KEY` (direct API) or `USE_BEDROCK=1` + AWS credentials "
            "(Amazon Bedrock) to enable this page. Example:\n\n"
            "`export ANTHROPIC_API_KEY=sk-ant-...`"
        )

    tab1, tab2, tab3 = st.tabs(
        ["💬 Ask a Workforce Question", "🔍 Explain Employee Risk", "📋 Retention Strategy Generator"]
    )

    with tab1:
        st.write("**Try asking:**")
        st.write(
            " · ".join(
                f"_{q}_" for q in [
                    "Which department needs attention?",
                    "Summarize our DEI representation across departments.",
                    "What training programs have the worst outcomes?",
                ]
            )
        )
        query = st.text_input("Your question", key="workforce_q")
        if st.button("Ask", key="ask_btn") and query:
            if not has_key:
                st.error("No LLM backend configured — see the warning above.")
            else:
                with st.spinner("Analyzing workforce data..."):
                    try:
                        st.markdown(answer_workforce_question(query))
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    with tab2:
        st.write("Ask why a specific employee is flagged as a risk — e.g. *\"Why is employee 1024 high risk?\"*")
        emp_id = st.number_input(
            "Employee ID", min_value=int(preds["EmployeeID"].min()),
            max_value=int(preds["EmployeeID"].max()), step=1,
        )
        if st.button("Explain", key="explain_btn"):
            if not has_key:
                st.error("No LLM backend configured — see the warning above.")
            else:
                with st.spinner(f"Analyzing employee {emp_id}..."):
                    try:
                        st.markdown(explain_employee_risk(int(emp_id)))
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    with tab3:
        st.write("Generate a retention action plan — for the whole workforce or one department.")
        dept_choice = st.selectbox(
            "Scope", ["Entire workforce"] + sorted(preds["DepartmentType"].unique().tolist())
        )
        if st.button("Generate Strategy", key="strategy_btn"):
            if not has_key:
                st.error("No LLM backend configured — see the warning above.")
            else:
                scope = None if dept_choice == "Entire workforce" else dept_choice
                with st.spinner("Generating retention strategy..."):
                    try:
                        st.markdown(generate_retention_strategy(department=scope))
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    st.divider()
    st.caption(
        "Each request sends only aggregate statistics (or one employee's own record, when "
        "explicitly asked about that employee) — never the full employee table."
    )

# ==================================================================== PAGE 10
elif page == "HR Policy Assistant (RAG)":
    from analytics.hr_knowledge_base import HRKnowledgeBase, answer_policy_question, POLICIES_DIR

    st.header("📚 HR Policy Assistant (RAG)")
    st.caption("Phase 12 · Ask questions answered directly from your uploaded HR policy documents")

    uploaded = st.file_uploader(
        "Upload HR policy documents (PDF or TXT)", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploaded:
        for f in uploaded:
            (POLICIES_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} document(s) to data/policies/")

    existing = sorted(p.name for p in POLICIES_DIR.glob("*") if p.suffix.lower() in (".pdf", ".txt", ".md"))
    if existing:
        st.write(f"**{len(existing)} document(s) available:** " + ", ".join(existing))
    else:
        st.info("No policy documents yet — upload one above (e.g. benefits, leave, or handbook PDFs).")

    question = st.text_input("Ask a policy question", placeholder="What benefits are available for employees?")
    if st.button("Ask Policy Question") and question:
        if not existing:
            st.error("Upload at least one policy document first.")
        elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("USE_BEDROCK") == "1"):
            st.error("No LLM backend configured — set ANTHROPIC_API_KEY or USE_BEDROCK=1.")
        else:
            with st.spinner("Searching policy documents..."):
                kb = HRKnowledgeBase().load()
                hits = kb.retrieve(question)
                if not hits:
                    st.warning("No relevant passages found in the uploaded documents.")
                else:
                    try:
                        st.markdown(answer_policy_question(kb, question))
                        with st.expander("Show retrieved source excerpts"):
                            for h in hits:
                                st.caption(f"**{h['source']}** (relevance: {h['score']})")
                                st.text(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))
                    except Exception as e:
                        st.error(f"Request failed: {e}")
