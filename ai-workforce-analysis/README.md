# AI-Powered Workforce Intelligence Platform Using AWS SageMaker, Amazon Bedrock & Predictive Analytics

End-to-end workforce analytics: data cleaning → EDA → attrition prediction (ML + SHAP)
→ interactive Streamlit dashboard with a Bedrock/Claude-powered AI HR assistant and
a RAG-based policy Q&A assistant. All 13 phases from the original roadmap are complete.

## Quick start (local, zero AWS setup required)

```bash
pip install -r requirements.txt
python run_pipeline.py        # Phases 2-9: clean, EDA, train models, score risk
streamlit run app.py          # Phase 8: launch the dashboard
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Deploy the dashboard

The dashboard can be deployed without S3, Athena, SageMaker, or Bedrock:
it already includes the processed CSVs and local trained model needed for a
working demonstration. The quickest public deployment is Streamlit Community
Cloud:

1. Create a GitHub repository and upload this project, including `data/`,
   `models/`, `app.py`, `requirements.txt`, and `.streamlit/config.toml`.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   choose the repository, branch, and `app.py`, then select **Deploy**.
3. Copy the URL Streamlit displays after deployment. This is your public
   project link.

For a container host, the supplied `Dockerfile` starts the app on port 8501:

```bash
docker build -t workforce-dashboard .
docker run --rm -p 8501:8501 workforce-dashboard
```

Do not publish the real employee data publicly. Restrict viewers or use an
authenticated hosting service when the data is not a classroom/demo dataset.

## Amazon S3 (optional, separate setup)

S3 is not required for the finished dashboard or its demo deployment. Use it
only when you want an AWS-backed data pipeline. Follow the concise Windows
steps in [`S3_SETUP.md`](S3_SETUP.md), then use `aws_setup_guide.md` for the
advanced Glue, Athena, SageMaker, and Bedrock setup.

To enable the AI Assistant pages (Phase 11-12), set an API key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Going further: live AWS backends (Phase 10-13)

Everything below is optional — the dashboard works fully without it, and
automatically switches to the live AWS backend once you configure it
(check the sidebar's "Backend status" panel to see which is active):

```bash
# Phase 10: use a real SageMaker endpoint instead of the local .pkl model
export SAGEMAKER_ENDPOINT_NAME=workforce-attrition-endpoint
export AWS_REGION=us-east-1

# Phase 11: use Amazon Bedrock instead of the direct Anthropic API
export USE_BEDROCK=1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

streamlit run app.py
```

Full step-by-step setup (S3, Glue, Athena, SageMaker training/registry/
endpoint, Bedrock, Bedrock Knowledge Base, ECS/Streamlit Cloud
deployment) is in **`aws_setup_guide.md`**.

## Project structure

```
ai-workforce-analysis/
├── data/
│   ├── raw/               your 3 source CSVs (employee_master, training_performance, recruitment)
│   ├── processed/         workforce_master.csv, employee_predictions.csv (generated)
│   └── policies/           HR policy docs for the RAG assistant (sample benefits/leave docs included)
├── analytics/
│   ├── data_pipeline.py    Phase 2-4: extract, clean, feature-engineer, merge
│   ├── eda.py               Phase 5: exploratory analysis + 6 chart reports
│   ├── attrition_model.py   Phase 6-7: train LR/RF/XGBoost, select best, save
│   ├── predict_risk.py      Phase 9: score workforce, SHAP reasons, recommendations
│   ├── risk_endpoint_client.py  Phase 10: SageMaker endpoint client w/ local fallback
│   ├── ai_hr_assistant.py    Phase 11: Bedrock/Claude HR assistant (3 features)
│   └── hr_knowledge_base.py  Phase 12: RAG over HR policy documents
├── aws/
│   ├── sagemaker_train.py         Phase 10: SageMaker training entrypoint (tested locally + in-container)
│   ├── sagemaker_inference.py     Phase 10: SageMaker endpoint inference handler
│   ├── launch_sagemaker_training.py  Phase 10: boto3/SageMaker SDK - launch training + register model
│   └── deploy_endpoint.py          Phase 10: boto3/SageMaker SDK - deploy approved model to an endpoint
├── models/                  saved model, features, scaler, comparison table (generated)
├── reports/                  6 PNG EDA visualizations (generated)
├── app.py                    Phase 8: Streamlit dashboard (10 pages, see below)
├── run_pipeline.py            runs data_pipeline → eda → attrition_model → predict_risk
├── aws_setup_guide.md          Phase 1-2 & 10-13: full AWS setup, SageMaker, Bedrock, RAG, deployment
└── requirements.txt
```

## What's in your data

- **employee_master.csv** (3,000 rows) — demographics, dept/division, employment status,
  termination info, tenure/dates. This is where the `Attrition` label comes from
  (`Voluntarily Terminated` / `Terminated for Cause` = left).
- **training_performance.csv** — satisfaction/engagement/work-life-balance scores,
  performance rating, training program history and outcomes.
- **recruitment.csv** — hiring-time data: education level, years of experience,
  desired salary, application status.

All three are merged on `EmployeeID` into `data/processed/workforce_master.csv`.

## Dashboard pages (app.py)

1. **Executive KPIs** — headcount, high-risk count, attrition rate, avg satisfaction/performance/tenure
2. **Predictive Attrition Risk** — risk category breakdown, probability distribution, top 15 at-risk employees
3. **Explainable AI (SHAP)** — model comparison table, global feature importance, most common risk drivers
4. **Department Risk Analysis** — attrition risk by department, heatmap
5. **Employee Analytics** — satisfaction/engagement/performance/tenure distributions, tenure-vs-satisfaction scatter colored by risk
6. **Skills & Career Development** — training outcomes, program participation, cost by type, education-vs-performance, training gaps list
7. **DEI Metrics** — gender/race/marital-status distribution, attrition by gender/race, department & pay-zone representation
8. **Employee Risk Directory** — searchable/filterable table with CSV export
9. **AI Workforce Assistant** *(Phase 11)* — 3 tabs: ask workforce questions, explain a specific employee's risk, generate a retention strategy — via Bedrock or direct Claude API
10. **HR Policy Assistant (RAG)** *(Phase 12)* — upload HR policy PDFs/TXT and ask questions answered only from those documents, with source excerpts shown

## Model performance (current dataset)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (selected) | 77.2% | 30.5% | 61.0% | 40.7% | **80.7%** |
| XGBoost | 78.0% | 28.7% | 48.1% | 35.9% | 80.2% |
| Logistic Regression | 72.2% | 27.5% | 71.4% | 39.7% | 76.5% |

Attrition is naturally imbalanced (~13% of the workforce), so all
models use class-balancing; Random Forest was selected for best
ROC-AUC (ability to rank employees by risk), which is what the
dashboard uses to prioritize retention outreach.

## Re-running after new data

Replace the files in `data/raw/` (or point `aws_setup_guide.md`'s
extractor at fresh Athena data) and re-run:

```bash
python run_pipeline.py
```

The dashboard will pick up the refreshed `data/processed/` files on
next launch (or click "Rerun" in Streamlit if it's already open).

## Phase completion status

| Phase | Status |
|---|---|
| 1-2: AWS S3 + Athena setup | ✅ Guide + boto3 extraction script (`aws_setup_guide.md`) |
| 3-6: Data engineering + EDA | ✅ `analytics/data_pipeline.py`, `eda.py` |
| 7: ML model training | ✅ `analytics/attrition_model.py` |
| 8: Streamlit dashboard | ✅ `app.py` (10 pages) |
| 9: Prediction engine | ✅ `analytics/predict_risk.py` |
| 10: SageMaker migration | ✅ `aws/sagemaker_train.py`, `sagemaker_inference.py`, `launch_sagemaker_training.py`, `deploy_endpoint.py`, `analytics/risk_endpoint_client.py` |
| 11: Bedrock AI HR Assistant | ✅ `analytics/ai_hr_assistant.py` + dashboard page |
| 12: RAG HR Knowledge Assistant | ✅ `analytics/hr_knowledge_base.py` + dashboard page |
| 13: Full AWS deployment | ✅ Guide (`aws_setup_guide.md`) — architecture, hosting, scheduling, team split |
