# Optional Amazon S3 setup

S3 is **not needed** to run or deploy the dashboard in demo mode. The project
already contains its processed data and trained model. Complete these steps
only if you want AWS to store the raw HR files for Athena, SageMaker, or a
production data-refresh workflow.

## 1. Prepare AWS locally

1. Create or sign in to your AWS account.
2. Create an IAM user (or use IAM Identity Center) with access limited to the
   new S3 bucket, Glue, and Athena. Do not use the root-account access keys.
3. Install the [AWS CLI](https://aws.amazon.com/cli/) and configure it in
   PowerShell:

   ```powershell
   aws configure
   aws sts get-caller-identity
   ```

   Choose one AWS region and use it consistently; `us-east-1` is used below.

## 2. Create the buckets

Choose a globally unique name, such as
`yourname-workforce-analytics-2026`. Do not make the bucket public because it
contains employee data.

```powershell
$bucket = "yourname-workforce-analytics-2026"
$resultsBucket = "yourname-workforce-athena-results-2026"
$region = "us-east-1"

aws s3api create-bucket --bucket $bucket --region $region
aws s3api create-bucket --bucket $resultsBucket --region $region
aws s3api put-public-access-block --bucket $bucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws s3api put-public-access-block --bucket $resultsBucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

If you use a region other than `us-east-1`, add
`--create-bucket-configuration LocationConstraint=$region` to each
`create-bucket` command.

## 3. Upload the three raw datasets

Run these commands from the project folder:

```powershell
aws s3 cp data/raw/employee_master.csv "s3://$bucket/raw/employee_master/employee_master.csv"
aws s3 cp data/raw/training_performance.csv "s3://$bucket/raw/training_performance/training_performance.csv"
aws s3 cp data/raw/recruitment.csv "s3://$bucket/raw/recruitment/recruitment.csv"
aws s3 ls "s3://$bucket/raw/" --recursive
```

Keep each CSV under its own prefix as shown. That lets Glue/Athena infer one
table per dataset.

## 4. Use it with the project (optional)

Set the Athena query-result location to
`s3://yourname-workforce-athena-results-2026/`, then continue at **Step 2** in
[`aws_setup_guide.md`](aws_setup_guide.md). The existing guide covers Glue,
Athena, SageMaker, Bedrock, and production hosting.

## Cost and security notes

- S3 storage is inexpensive, but Athena charges per data scanned and SageMaker
  endpoints incur hourly charges while running. Delete/stop unused endpoints.
- Enable default bucket encryption (SSE-S3 is a reasonable default) and versioning
  if this will contain real HR records.
- Do not put AWS access keys, employee exports, or uploaded confidential policy
  PDFs in GitHub or in a public Streamlit deployment.
