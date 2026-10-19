# Reproduce it with AWS Bedrock batch inference

Bedrock batch inference runs a JSONL file of requests asynchronously at roughly half the on-demand price. This repo
wraps it: upload, submit, poll, download, and re-attach if your terminal dies.

## One-time setup

1. In the Bedrock console, **request model access** for your chosen Claude model in your region.
2. `pip install -e ".[aws,embed,map]"` and configure AWS credentials (`aws configure sso` or environment variables).
3. `AWS_REGION=us-east-1 ./scripts/aws_setup.sh` creates a private, encrypted S3 bucket (`abstract-atlas-batch-<account>`,
   30-day expiry) and an IAM role (`abstract-atlas-bedrock-batch`) that Bedrock can assume to read/write that bucket
   and invoke models. It prints the extra permissions your own identity needs (`bedrock:CreateModelInvocationJob`,
   `iam:PassRole`, ...).

The scripts' defaults are discovered from your account id; override with `ATLAS_BATCH_BUCKET` / `ATLAS_BATCH_ROLE`
or `bucket:` / `role_arn:` in the backend config.

## Config

```yaml
backends:
  bedrock_batch: {type: bedrock-batch, model: us.anthropic.claude-sonnet-4-6, region: us-east-1}
  bedrock:       {type: bedrock,       model: us.anthropic.claude-sonnet-4-6, region: us-east-1}
use: {default: bedrock_batch, naming: bedrock}   # cluster naming is interactive, so on-demand
```

Model IDs vary by region and change over time; use an ID your account lists under Bedrock -> Inference profiles.

## Run

```bash
atlas -c my.yaml estimate --price-in <USD/Mtok> --price-out <USD/Mtok>   # check before you spend
atlas -c my.yaml enrich      # submits a batch job, polls, writes enrichment.jsonl
atlas -c my.yaml audit       # a second batch job
atlas -c my.yaml run --skip enrich,audit   # embed, topics, trends, map
```

Batch jobs can take minutes to hours. If you interrupt `atlas`, run the same command again: `runs/<name>/batch_jobs.json`
records the job, and a rerun re-attaches as long as the records and prompts are unchanged (it fingerprints both).

## Things that bit us

- **Minimum 100 records per job.** Smaller workloads automatically run on-demand.
- **Not every model supports batch.** If Bedrock rejects the model for batch, the backend warns and falls back to on-demand.
- **Some models are slow to start.** In the original account, one smaller model sat "Scheduled" for 3+ hours while a larger one started in minutes. Try another model before assuming something is broken.
- **Forced tool calls.** Some newer Claude models reject `tool_choice: tool`. The backend probes once per model and falls back to `auto` plus an instruction.
- **Corporate TLS inspection** can break boto3 or Hugging Face downloads; set `AWS_CA_BUNDLE` / `REQUESTS_CA_BUNDLE` to your proxy's CA.
- **Cleanup.** The bucket expires objects after 30 days. Delete the role and bucket when you are done with the project.

Not covered by automated tests: the batch path needs real AWS and was not exercised by this repo's test suite. The
request/response handling is unit-tested; the job lifecycle is a port of code that ran the NeurIPS and SCALE builds.
