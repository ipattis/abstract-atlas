"""Claude via the Anthropic API or AWS Bedrock: on-demand calls, plus Bedrock batch inference (~50% cheaper).

Structured output uses a single forced tool call. Some newer Claude models reject forced `tool_choice`, so
the first call probes the model once and falls back to `tool_choice=auto` plus an instruction.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import Backend

MIN_BATCH_RECORDS = 100  # Bedrock's minimum per batch job; smaller workloads run on-demand
TERMINAL = {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired"}


def tool_body(req, forced):
    tool = {"name": req.name, "description": req.description, "input_schema": req.schema}
    body = {"max_tokens": req.max_tokens, "tools": [tool], "messages": [{"role": "user", "content": req.prompt}]}
    system = req.system
    if forced:
        body["tool_choice"] = {"type": "tool", "name": req.name}
    else:
        body["tool_choice"] = {"type": "auto"}
        system = (system + "\n\n" if system else "") + f"Respond only by calling the {req.name} tool."
    if system:
        body["system"] = system
    return body


def parse_response(resp):
    """Messages API response (dict) -> result dict."""
    if "error" in resp:
        return {"error": str(resp["error"])[:300]}
    for block in resp.get("content", []):
        if block.get("type") == "tool_use":
            usage = resp.get("usage") or {}
            return {"output": block["input"],
                    "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}
    return {"error": "no tool_use block in response"}


class ClaudeBackend(Backend):
    """On-demand Claude. `provider: anthropic` uses ANTHROPIC_API_KEY; `provider: bedrock` uses AWS credentials."""

    def __init__(self, model, provider="bedrock", region=None, workers=8, state_dir=".", **_):
        self.model, self.provider, self.workers = model, provider, workers
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.state_dir = Path(state_dir)
        self.name = f"claude:{provider}:{model}"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            if self.provider == "anthropic":
                self._client = anthropic.Anthropic(max_retries=10)
            else:
                self._client = anthropic.AnthropicBedrock(aws_region=self.region, max_retries=10)
        return self._client

    def forced_tools_ok(self):
        import anthropic

        path = self.state_dir / "model_caps.json"
        caps = json.loads(path.read_text()) if path.exists() else {}
        if self.model not in caps:
            tool = {"name": "ping", "description": "ping", "input_schema": {"type": "object", "properties": {}}}
            try:
                self.client.messages.create(model=self.model, max_tokens=16, tools=[tool],
                                            tool_choice={"type": "tool", "name": "ping"},
                                            messages=[{"role": "user", "content": "ping"}])
                caps[self.model] = True
            except anthropic.BadRequestError as e:
                if "tool_choice" not in str(e):
                    raise
                caps[self.model] = False
            path.write_text(json.dumps(caps, indent=2))
        return caps[self.model]

    def bodies(self, requests):
        forced = self.forced_tools_ok()
        return {rid: tool_body(req, forced) for rid, req in requests.items()}

    def _on_demand(self, bodies):
        def one(item):
            rid, body = item
            try:
                return rid, parse_response(self.client.messages.create(model=self.model, **body).model_dump())
            except Exception as e:
                return rid, {"error": repr(e)[:300]}

        with ThreadPoolExecutor(self.workers) as pool:
            return dict(pool.map(one, bodies.items()))

    def run(self, requests, job="job"):
        return self._on_demand(self.bodies(requests)) if requests else {}

    def chat_text(self, system, user, max_tokens=1024):
        kwargs = {"system": system} if system else {}
        resp = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": user}], **kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")


class BedrockBatchBackend(ClaudeBackend):
    """Bedrock batch inference: upload JSONL to S3, submit a job, poll, download.

    Job state is saved in <state_dir>/batch_jobs.json, so an interrupted run re-attaches to the same job instead
    of paying twice. A job is reused only if its records AND prompts are unchanged.

    Needs an S3 bucket and an IAM role Bedrock can assume (see scripts/aws_setup.sh and docs/aws-batch.md).
    """

    def __init__(self, model, bucket=None, role_arn=None, prefix="abstract-atlas", poll_seconds=60, **kw):
        super().__init__(model, provider="bedrock", **kw)
        self.bucket, self.role_arn, self.prefix, self.poll_seconds = bucket, role_arn, prefix, poll_seconds
        self.name = f"bedrock-batch:{model}"
        self.state_path = self.state_dir / "batch_jobs.json"

    def _state(self):
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _infra(self):
        import boto3

        account = boto3.client("sts").get_caller_identity()["Account"]
        return (self.bucket or os.environ.get("ATLAS_BATCH_BUCKET") or f"abstract-atlas-batch-{account}",
                self.role_arn or os.environ.get("ATLAS_BATCH_ROLE") or f"arn:aws:iam::{account}:role/abstract-atlas-bedrock-batch")

    @staticmethod
    def fingerprint(bodies):
        h = hashlib.sha256()
        for rid in sorted(bodies):
            h.update(rid.encode())
            h.update(json.dumps(bodies[rid], sort_keys=True).encode())
        return h.hexdigest()[:16]

    def run(self, requests, job="job", ):
        if not requests:
            return {}
        bodies = self.bodies(requests)
        if len(bodies) < MIN_BATCH_RECORDS:
            print(f"[{job}] {len(bodies)} records < {MIN_BATCH_RECORDS}; using on-demand calls")
            return self._on_demand(bodies)

        import boto3

        bedrock = boto3.client("bedrock", region_name=self.region)
        s3 = boto3.client("s3", region_name=self.region)
        bucket, role = self._infra()
        state = self._state()
        fp = self.fingerprint(bodies)
        rec = state.get(job)
        if rec and rec.get("fingerprint") != fp:
            print(f"[{job}] records or prompts changed since the last submission; submitting a new job")
            rec = None
        if rec:
            status = bedrock.get_model_invocation_job(jobIdentifier=rec["arn"])["status"]
            if status in {"Failed", "Stopped", "Expired"}:
                print(f"[{job}] previous job {status}; resubmitting")
                rec = None

        if not rec:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            prefix = f"{self.prefix}/{job}/{stamp}"
            lines = [json.dumps({"recordId": rid, "modelInput": {"anthropic_version": "bedrock-2023-05-31", **body}})
                     for rid, body in bodies.items()]
            s3.put_object(Bucket=bucket, Key=f"{prefix}/input.jsonl", Body="\n".join(lines).encode())
            try:
                resp = bedrock.create_model_invocation_job(
                    jobName=f"{job}-{stamp}"[:63], roleArn=role, modelId=self.model,
                    inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{bucket}/{prefix}/input.jsonl"}},
                    outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{bucket}/{prefix}/output/"}},
                    timeoutDurationInHours=24,
                )
            except bedrock.exceptions.ValidationException as e:
                if "not supported for the requested model" not in str(e):
                    raise
                print(f"[{job}] WARNING: {self.model} doesn't support batch inference; using on-demand calls")
                return self._on_demand(bodies)
            rec = {"arn": resp["jobArn"], "prefix": prefix, "fingerprint": fp}
            state[job] = rec
            self.state_path.write_text(json.dumps(state, indent=2))
            print(f"[{job}] submitted {len(bodies)} records as {resp['jobArn'].split('/')[-1]}")

        start = time.time()
        while True:
            info = bedrock.get_model_invocation_job(jobIdentifier=rec["arn"])
            if info["status"] in TERMINAL:
                break
            print(f"[{job}] {info['status']} ({(time.time() - start) / 60:.0f} min)", flush=True)
            time.sleep(self.poll_seconds)
        if info["status"] not in {"Completed", "PartiallyCompleted"}:
            raise RuntimeError(f"[{job}] batch job {info['status']}: {info.get('message')}")

        job_id = rec["arn"].split("/")[-1]
        body = s3.get_object(Bucket=bucket, Key=f"{rec['prefix']}/output/{job_id}/input.jsonl.out")["Body"].read().decode()
        results = {}
        for line in body.splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["recordId"]] = parse_response(row.get("modelOutput") or {"error": row.get("error", "missing output")})
        n_err = sum("error" in r for r in results.values())
        print(f"[{job}] {info['status']}: {len(results) - n_err} ok, {n_err} errors, {len(bodies) - len(results)} missing")
        return results
