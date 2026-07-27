"""
Phase 10: Launch a SageMaker Training Job + Register in Model Registry
--------------------------------------------------------------------------
Run this from your own machine/AWS Cloud9 with AWS credentials configured
(`aws configure`) and the `sagemaker` SDK installed (`pip install sagemaker`).
I can't execute this from this environment - no AWS access here - so treat
it as a ready-to-run script, not something already run on your behalf.

Prerequisites:
  - workforce_master.csv uploaded to S3 (see aws_setup_guide.md Phase 1-2)
  - aws/sagemaker_train.py and aws/sagemaker_inference.py in this same folder
  - An IAM execution role for SageMaker (SageMakerExecutionRole) with S3 read/write
"""

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model

ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
BUCKET = "your-workforce-analytics-bucket"
REGION = "us-east-1"

session = sagemaker.Session()

# --------------------------------------------------------------- Training
sklearn_estimator = SKLearn(
    entry_point="sagemaker_train.py",
    source_dir="aws",  # contains sagemaker_train.py
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={"n-estimators": 300, "max-depth": 8},
    metric_definitions=[
        {"Name": "rf:roc_auc", "Regex": r"model=RandomForest.*roc_auc=([0-9\.]+)"},
        {"Name": "xgb:roc_auc", "Regex": r"model=XGBoost.*roc_auc=([0-9\.]+)"},
    ],
)

train_input = f"s3://{BUCKET}/processed/workforce_master.csv"
print("Starting SageMaker training job...")
sklearn_estimator.fit({"train": f"s3://{BUCKET}/processed/"})
print(f"Training job complete. Model artifact: {sklearn_estimator.model_data}")

# --------------------------------------------------------------- Model Registry
model_package_group = "workforce-attrition-models"

model = Model(
    image_uri=sklearn_estimator.image_uri,
    model_data=sklearn_estimator.model_data,
    role=ROLE_ARN,
    entry_point="sagemaker_inference.py",
    source_dir="aws",
)

model_package = model.register(
    model_package_group_name=model_package_group,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large", "ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    approval_status="PendingManualApproval",  # approve in console/CLI before deploying
)

print(f"Registered model version: {model_package.model_package_arn}")
print("Next: approve this version in the SageMaker console, then run deploy_endpoint.py")
