# Phase 1-2: AWS Athena Setup & Python Integration

This guide covers wiring your HR data source up to AWS so the same
pipeline (`analytics/data_pipeline.py`) can pull live data instead of
local CSVs. Run these steps in your own AWS account/CLI — I can't
provision AWS resources from this environment, so nothing here has
been executed on your behalf.

## Prerequisites
- An AWS account with permissions for S3, Glue, and Athena
- AWS CLI installed and configured (`aws configure`)
- Your three HR CSVs: `employee_master.csv`, `training_performance.csv`, `recruitment.csv`

## Step 1: Create an S3 bucket and upload the data

```bash
aws s3 mb s3://your-workforce-analytics-bucket --region us-east-1

aws s3 cp data/raw/employee_master.csv \
  s3://your-workforce-analytics-bucket/raw/employee_master/employee_master.csv
aws s3 cp data/raw/training_performance.csv \
  s3://your-workforce-analytics-bucket/raw/training_performance/training_performance.csv
aws s3 cp data/raw/recruitment.csv \
  s3://your-workforce-analytics-bucket/raw/recruitment/recruitment.csv
```

Note each dataset gets its own "folder" (prefix) — Glue/Athena expect
one table per prefix, not one prefix with mixed schemas.

## Step 2: Create a Glue Data Catalog database

```bash
aws glue create-database --database-input '{"Name": "workforce_analytics_db"}'
```

## Step 3: Define table schemas (Glue Data Catalog)

Run a Glue Crawler (easiest) or define tables manually. Crawler approach:

```bash
aws glue create-crawler \
  --name workforce-analytics-crawler \
  --role YOUR_GLUE_SERVICE_ROLE_ARN \
  --database-name workforce_analytics_db \
  --targets '{"S3Targets": [{"Path": "s3://your-workforce-analytics-bucket/raw/"}]}'

aws glue start-crawler --name workforce-analytics-crawler
```

This auto-detects columns/types for all three CSVs and registers three
tables (`employee_master`, `training_performance`, `recruitment`) in
`workforce_analytics_db`.

## Step 4: Configure an Athena query results location

Athena needs an S3 path to write query results to:

```bash
aws s3 mb s3://your-workforce-analytics-bucket-athena-results --region us-east-1
```

In the Athena console (or via `aws athena start-query-execution`), set
this as your "Query result location" under Workgroup settings.

## Step 5: Verify with a test query

```sql
SELECT COUNT(*) AS row_count FROM workforce_analytics_db.employee_master;

SELECT DepartmentType, COUNT(*) AS headcount
FROM workforce_analytics_db.employee_master
GROUP BY DepartmentType
ORDER BY headcount DESC;
```

## Step 6 (Phase 2): Connect from Python via boto3

Install dependencies:

```bash
pip install boto3 pandas
```

`analytics/aws_athena_extract.py` (new file to add if you go this
route — drop it in `analytics/` alongside `data_pipeline.py`):

```python
"""
Pulls the three HR tables from Athena into local CSVs, replacing the
data/raw/ files used by data_pipeline.py. Requires AWS credentials
configured via `aws configure` or environment variables.
"""

import time
import boto3
import pandas as pd
from pathlib import Path

DATABASE = "workforce_analytics_db"
S3_OUTPUT = "s3://your-workforce-analytics-bucket-athena-results/"
REGION = "us-east-1"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

athena = boto3.client("athena", region_name=REGION)


def run_athena_query(sql: str) -> pd.DataFrame:
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
    )
    query_id = response["QueryExecutionId"]

    # Poll until the query finishes
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query failed: {reason}")

    # Read results directly from the S3 output location
    result_path = status["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    return pd.read_csv(result_path)


def extract_all_tables():
    tables = {
        "employee_master": "SELECT * FROM employee_master",
        "training_performance": "SELECT * FROM training_performance",
        "recruitment": "SELECT * FROM recruitment",
    }
    for name, sql in tables.items():
        print(f"Querying {name}...")
        df = run_athena_query(sql)
        out_path = RAW_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> saved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    extract_all_tables()
```

After running this script, `data/raw/*.csv` will contain fresh data
pulled live from Athena, and you can run the rest of the pipeline
exactly as before:

```bash
python analytics/aws_athena_extract.py   # Phase 1-2: live pull from Athena
python run_pipeline.py                   # Phase 3-9: clean, EDA, train, score
streamlit run app.py                     # Phase 8: dashboard
```

## Production notes (Phase 10-13 covered below in detail)

- **Scheduled refresh:** use EventBridge (cron rule) to trigger a Lambda
  that runs `aws_athena_extract.py` + `run_pipeline.py` on a schedule
  (e.g., nightly), or orchestrate the same steps with AWS Step Functions.
- **Model retraining:** move `attrition_model.py` into a SageMaker
  Processing/Training job if you want managed retraining infra instead
  of a Lambda with a time limit.
- **Dashboard hosting:** deploy `app.py` to Streamlit Community Cloud
  (simplest) or containerize it and run on AWS ECS/Fargate behind an ALB
  if it needs to live inside your VPC.
- **Credentials:** never hardcode AWS keys in the scripts above — use
  an IAM role (Lambda/ECS execution role) in production, and
  `aws configure` / environment variables only for local dev.

---

# Phase 10: Migrate ML Pipeline to Amazon SageMaker

Files: `aws/sagemaker_train.py`, `aws/sagemaker_inference.py`,
`aws/launch_sagemaker_training.py`, `aws/deploy_endpoint.py`,
`analytics/risk_endpoint_client.py`.

`sagemaker_train.py` and `sagemaker_inference.py` are written to the
SageMaker Script Mode / inference contract, and have already been
tested locally in this project (they produce identical metrics to
`attrition_model.py`). They run unmodified inside a real SageMaker
Training Job / endpoint container — nothing to change.

## 1. Install the SageMaker SDK locally

```bash
pip install sagemaker boto3
```

## 2. Upload the training entrypoint + processed data to S3

```bash
aws s3 cp data/processed/workforce_master.csv \
  s3://your-workforce-analytics-bucket/processed/workforce_master.csv
```

## 3. Create a SageMaker execution role (one-time, via console or CLI)

IAM role with `AmazonSageMakerFullAccess` + S3 read/write to your bucket.
Note the Role ARN — you'll paste it into the scripts below.

## 4. Launch the training job & register the model

Edit `ROLE_ARN` and `BUCKET` at the top of `aws/launch_sagemaker_training.py`,
then:

```bash
python aws/launch_sagemaker_training.py
```

This trains Logistic Regression, Random Forest, and XGBoost as a managed
SageMaker Training Job, logs ROC-AUC/accuracy/precision/recall/F1 to
CloudWatch, and registers the resulting model artifact as a new version
in the SageMaker **Model Registry** (`workforce-attrition-models` group).

## 5. Approve the model version

In the SageMaker console → Model Registry → approve the version you want
to deploy (or `aws sagemaker update-model-package --model-approval-status Approved`).

## 6. Deploy the endpoint

Edit `MODEL_PACKAGE_ARN` in `aws/deploy_endpoint.py`, then:

```bash
python aws/deploy_endpoint.py
```

This creates a real-time endpoint matching the exact request/response
contract from the spec:

```json
// Request
{"satisfaction": 2, "tenure": 1.5, "department": "Sales"}
// Response
{"risk": "High", "probability": 0.87}
```

## 7. Point the dashboard at the live endpoint

```bash
export SAGEMAKER_ENDPOINT_NAME=workforce-attrition-endpoint
export AWS_REGION=us-east-1
streamlit run app.py
```

`analytics/risk_endpoint_client.py` automatically switches from the
local `.pkl` model to this live endpoint when `SAGEMAKER_ENDPOINT_NAME`
is set — no other code changes needed. The sidebar's "Backend status"
panel confirms which one is active.

---

# Phase 11: Amazon Bedrock AI HR Assistant

File: `analytics/ai_hr_assistant.py` (already built, tested, and wired
into the "AI Workforce Assistant" dashboard page with 3 tabs: Ask a
Workforce Question, Explain Employee Risk, Retention Strategy Generator).

## Local mode (works today, no AWS needed)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Bedrock mode (production)

1. In the Bedrock console, request access to an Anthropic Claude model
   (e.g. Claude 3.5 Sonnet) if not already enabled for your account.
2. Ensure your AWS credentials/IAM role include `bedrock:InvokeModel`.
3. Set:

```bash
export USE_BEDROCK=1
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
streamlit run app.py
```

`ai_hr_assistant.py` checks `USE_BEDROCK` and routes every call through
`bedrock-runtime.invoke_model` instead of the direct Anthropic API —
same prompts, same 3 features, same output format either way.

---

# Phase 12: RAG-Based HR Knowledge Assistant (Optional Advanced)

File: `analytics/hr_knowledge_base.py`, dashboard page **"HR Policy
Assistant (RAG)"**.

The version in this project uses TF-IDF retrieval (scikit-learn) so it
works fully offline — already tested against sample benefits/leave
policy docs in `data/policies/`. Use the dashboard's file uploader to
add your own HR Policies PDF, Benefits Document, Leave Policy, or
Employee Handbook, then ask questions like *"What benefits are
available for employees?"* — answers are grounded only in the uploaded
text, with source excerpts shown for verification.

## Upgrading retrieval to Amazon Bedrock Knowledge Bases (production)

1. Upload your policy PDFs to an S3 prefix, e.g.
   `s3://your-workforce-analytics-bucket/hr-policies/`.
2. In the Bedrock console, create a **Knowledge Base** pointing at that
   S3 prefix (Bedrock handles chunking + embedding + vector storage
   automatically via OpenSearch Serverless).
3. Replace `HRKnowledgeBase.retrieve()` in `hr_knowledge_base.py` with:

```python
def retrieve(self, query, top_k=4):
    import boto3
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    response = client.retrieve(
        knowledgeBaseId="YOUR_KB_ID",
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
    )
    return [
        {"text": r["content"]["text"], "source": r["location"]["s3Location"]["uri"], "score": r["score"]}
        for r in response["retrievalResults"]
    ]
```

Everything downstream (`answer_policy_question`, the dashboard page)
stays identical.

---

# Phase 13: Deploy the Complete AWS Application

Final architecture (all pieces already built in this project):

```
S3 (raw HR data)
  -> Athena (query layer)
  -> SageMaker (training + endpoint)         [Phase 10]
  -> Employee Risk Predictions
       -> Streamlit Dashboard                [Phase 8]
       -> Bedrock AI Assistant                [Phase 11]
       -> Bedrock Knowledge Base (optional)    [Phase 12]
  -> HR Decision Support
```

## Deployment checklist

1. **Data & models:** S3 buckets, Glue Catalog, Athena, SageMaker
   endpoint all live (Phases 1-2, 10).
2. **Dashboard hosting** — two good options:
   - **Streamlit Community Cloud** (fastest): push this repo to GitHub,
     connect it at [share.streamlit.io](https://share.streamlit.io),
     and set `SAGEMAKER_ENDPOINT_NAME`, `USE_BEDROCK`, `AWS_REGION`,
     and AWS credentials as app secrets.
   - **AWS ECS/Fargate** (if it must live inside your VPC): containerize
     with a `Dockerfile` (`FROM python:3.11-slim`, `pip install -r
     requirements.txt`, `CMD ["streamlit","run","app.py"]`), push to ECR,
     run as an ECS service behind an Application Load Balancer, and grant
     the task role `sagemaker:InvokeEndpoint` + `bedrock:InvokeModel`.
3. **Scheduled retraining:** EventBridge rule -> Step Functions state
   machine that re-runs `aws_athena_extract.py` -> `launch_sagemaker_training.py`
   -> re-approves and re-deploys the endpoint on a schedule (e.g. weekly).
4. **Access control:** put the dashboard behind your company SSO
   (e.g. an ALB with Cognito authentication, or Streamlit Community
   Cloud's built-in viewer restrictions) since it surfaces individual
   employee risk scores.

## Suggested team split (from the original brief)

| Person | Focus | Already built for you |
|---|---|---|
| 1 | AWS + SageMaker | `aws/sagemaker_train.py`, `launch_sagemaker_training.py`, `deploy_endpoint.py` — just fill in your ARNs and run |
| 2 | Bedrock Integration | `analytics/ai_hr_assistant.py` — set `USE_BEDROCK=1` and your model ID |
| 3 | Dashboard Enhancement | `app.py` — AI Assistant + RAG pages, backend-status indicator already added |
| 4 | Documentation | This guide + `README.md` — add your architecture diagram and deployment specifics |

