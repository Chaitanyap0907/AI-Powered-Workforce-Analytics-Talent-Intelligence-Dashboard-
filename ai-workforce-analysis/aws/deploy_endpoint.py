"""
Phase 10: Deploy an Approved Model to a Real-Time SageMaker Endpoint
--------------------------------------------------------------------------
Run after launch_sagemaker_training.py and after approving the model
version in the SageMaker console (or via boto3 update_model_package).
Not executed here - run this yourself with AWS credentials configured.
"""

import boto3
import sagemaker
from sagemaker import ModelPackage

ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:YOUR_ACCOUNT_ID:model-package/workforce-attrition-models/1"
ENDPOINT_NAME = "workforce-attrition-endpoint"

session = sagemaker.Session()

model = ModelPackage(
    role=ROLE_ARN,
    model_package_arn=MODEL_PACKAGE_ARN,
    sagemaker_session=session,
)

print(f"Deploying endpoint: {ENDPOINT_NAME} ...")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=ENDPOINT_NAME,
)
print(f"Endpoint deployed: {ENDPOINT_NAME}")

# ------------------------------------------------------- Quick smoke test
sample_request = {"satisfaction": 2, "tenure": 1.5, "department": "Sales"}
result = predictor.predict(sample_request)
print(f"Sample prediction: {result}")

print(
    f"\nSet this in your dashboard environment to switch the app from the "
    f"local .pkl model to this live endpoint:\n\n"
    f"  export SAGEMAKER_ENDPOINT_NAME={ENDPOINT_NAME}\n"
    f"  export AWS_REGION=us-east-1\n"
)
