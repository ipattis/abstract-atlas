#!/usr/bin/env bash
# One-time AWS setup for Bedrock batch inference: an S3 bucket and an IAM role that Bedrock can assume.
#   AWS_REGION=us-east-1 ./scripts/aws_setup.sh
# Prerequisites: AWS CLI configured with permission to create buckets/roles, and Bedrock model access granted
# for your chosen Claude model in the console (Bedrock -> Model access). Safe to re-run.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${ATLAS_BATCH_BUCKET:-abstract-atlas-batch-${ACCOUNT}}"
ROLE="${ATLAS_BATCH_ROLE_NAME:-abstract-atlas-bedrock-batch}"

echo "Account $ACCOUNT, region $REGION, bucket $BUCKET, role $ROLE"

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
# Batch inputs/outputs are disposable once you've downloaded the results.
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-batch-files","Status":"Enabled","Filter":{"Prefix":""},"Expiration":{"Days":30}}]}'

TRUST="$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock.amazonaws.com"},
 "Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":"$ACCOUNT"},
 "ArnLike":{"aws:SourceArn":"arn:aws:bedrock:$REGION:$ACCOUNT:model-invocation-job/*"}}}]}
JSON
)"
POLICY="$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
  "Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Effect":"Allow","Action":["bedrock:InvokeModel"],
  "Resource":["arn:aws:bedrock:*::foundation-model/*","arn:aws:bedrock:*:$ACCOUNT:inference-profile/*"]}]}
JSON
)"

if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "$TRUST"
else
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" >/dev/null
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name batch-access --policy-document "$POLICY"

cat <<MSG

Done. Either rely on the default names (picked up automatically from your account id) or set:
  export ATLAS_BATCH_BUCKET=$BUCKET
  export ATLAS_BATCH_ROLE=arn:aws:iam::$ACCOUNT:role/$ROLE
Your own identity also needs: bedrock:CreateModelInvocationJob, bedrock:GetModelInvocationJob,
s3:PutObject/GetObject on the bucket, iam:PassRole on the role, and bedrock:InvokeModel.
MSG
