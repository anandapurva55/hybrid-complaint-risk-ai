# AWS Deployment Notes: Hybrid Complaint Risk API

This documents an actual, live deployment of the FastAPI wrapper ([`app.py`](app.py), containerised through [`Dockerfile`](Dockerfile)) to **AWS App Runner**, performed end to end on 2026-06-07: building the image, pushing it to ECR, creating IAM roles, storing the API key in Secrets Manager, creating the App Runner service, testing it live, and then tearing everything back down.

## 1. Why App Runner

App Runner was chosen over SageMaker or Lambda for the following reasons:

- **Lambda** has a roughly 250 MB unzipped deployment-package limit (even with layers) and a hard execution-time ceiling. This project's image bundles TensorFlow plus three trained models (about 1.7 GB), and a single LLM-inclusive request already takes several seconds: a poor fit for Lambda's packaging model and cold-start behaviour.
- **SageMaker** endpoints are built around a specific model-serving contract (`/invocations`, `/ping`, `model.tar.gz` artifacts) and are priced for always-on machine learning serving infrastructure: considerably more machinery than a single FastAPI service like this one needs.
- **App Runner** runs an arbitrary container image straight from ECR, handles load balancing, TLS, scaling, and health checks automatically, and is billed per vCPU-second and memory-second while running. It is the simplest "give it a container, get a public HTTPS URL" path for a service like this.

## 2. Architecture (as deployed)

```
 Client
   │  HTTPS
   ▼
AWS App Runner  (hybrid-risk-api, 2 vCPU / 4 GB, public URL, automatic TLS)
   │  pulls the image from
   ▼
Amazon ECR  (private repository: hybrid-risk-api:latest, linux/amd64)
   │  the container is built from the Dockerfile and runs `uvicorn app:app`;
   │  on startup it reads GEMINI_API_KEY, supplied from
   ▼
AWS Secrets Manager  (hybrid-risk-api/GEMINI_API_KEY)
   │  injected as a runtime environment variable through RuntimeEnvironmentSecrets
   ▼
 [container] FastAPI -> hybrid_inference.predict_hybrid()
   ├─ classical_model.pkl / cnn_model.keras / rnn_model.keras  (loaded once, bundled in the image)
   └─ llm_explainer.explain_with_gemini()  -> the Gemini API (gemini-2.5-flash)
```

This is the same pipeline shown in the architecture diagram in [`README.md`](README.md), System architecture: the box labelled "container" above is exactly that pipeline, simply running in the cloud rather than on a laptop. The infrastructure around it exists purely to get HTTPS traffic to that container securely and to hand it the one secret it needs at startup.

Two IAM roles support this, each scoped to the minimum it needs:

- `AppRunnerECRAccessRole-hybrid-risk`, trusted by `build.apprunner.amazonaws.com`, carries the AWS-managed `AWSAppRunnerServicePolicyForECRAccess` policy, so that App Runner can pull the private image at deploy and build time.
- `AppRunnerInstanceRole-hybrid-risk`, trusted by `tasks.apprunner.amazonaws.com`, carries an inline policy granting `secretsmanager:GetSecretValue` on only the one Gemini-key secret's ARN, so that the running container can fetch that key, and nothing else in the account.

## 3. How the API key was kept secure

The Gemini key was never placed in the Dockerfile, the image, the App Runner service configuration, environment-variable plaintext, or this conversation:

1. It lived only in the local, git-ignored `.env` file (see the Setup section of [`README.md`](README.md) and [`.env.example`](.env.example)).
2. It was read once through `python-dotenv`, which correctly strips the quotes the `.env` file wraps it in (see the related gotcha in step 5 of the walkthrough below), written to a temporary file with permissions set to `chmod 600`, and passed to:
   ```
   aws secretsmanager create-secret \
     --name hybrid-risk-api/GEMINI_API_KEY \
     --secret-string file:///tmp/<tempfile>
   ```
   After this command ran, the temporary file was deleted (`trap 'rm -f "$TMPSECRET"' EXIT`). The key value never appeared in any command's visible output, in `~/.aws/*`, or anywhere in this conversation.
3. The App Runner service references the secret by its ARN, in `ImageConfiguration.RuntimeEnvironmentSecrets`, so App Runner injects the real value as `GEMINI_API_KEY` into the container's environment at startup. [`llm_explainer.py`](llm_explainer.py)'s existing `os.environ.get("GEMINI_API_KEY")` and `load_dotenv()` code required zero changes to run in the cloud.

## 4. Step by step: what was actually run

```bash
# 1. Create the ECR repository
aws ecr create-repository --repository-name hybrid-risk-api --region us-east-1 \
  --image-scanning-configuration scanOnPush=true

# 2. Authenticate Docker to ECR, build for linux/amd64 (App Runner's
#    architecture; the build machine is Apple Silicon / arm64, so a plain
#    `docker build` would have produced an incompatible image), and push
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 767397972889.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -t 767397972889.dkr.ecr.us-east-1.amazonaws.com/hybrid-risk-api:latest --load .
docker push 767397972889.dkr.ecr.us-east-1.amazonaws.com/hybrid-risk-api:latest

# 3. IAM role so that App Runner's build step can pull the private image
aws iam create-role --role-name AppRunnerECRAccessRole-hybrid-risk \
  --assume-role-policy-document file://<trust policy: build.apprunner.amazonaws.com>
aws iam attach-role-policy --role-name AppRunnerECRAccessRole-hybrid-risk \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess

# 4. Secret, plus the IAM role so the running container can read only that secret
aws secretsmanager create-secret --name hybrid-risk-api/GEMINI_API_KEY \
  --secret-string file:///tmp/<tempfile with the key>      # see section 3
aws iam create-role --role-name AppRunnerInstanceRole-hybrid-risk \
  --assume-role-policy-document file://<trust policy: tasks.apprunner.amazonaws.com>
aws iam put-role-policy --role-name AppRunnerInstanceRole-hybrid-risk \
  --policy-name read-gemini-secret --policy-document file://<policy: GetSecretValue on the one ARN>

# 5. Create the service
aws apprunner create-service --cli-input-json file://apprunner-service.json
#   (ImageRepository points at the ECR image, Port is 8000,
#    RuntimeEnvironmentSecrets maps GEMINI_API_KEY to the secret's ARN,
#    InstanceConfiguration specifies 2 vCPU / 4 GB plus
#    AppRunnerInstanceRole-hybrid-risk, and HealthCheckConfiguration
#    checks GET / on port 8000)
```

Result:
```
ServiceUrl: p6si42spqz.us-east-1.awsapprunner.com
Status: RUNNING (after roughly 10 minutes: about a 1-second image pull,
                 instance provisioning, and then App Runner's status field
                 lagged the container's actual readiness by several minutes;
                 see "Gotchas" below)
```

## 5. Live test: a sample request and response

The same complaint (`C002`, from [`sample_inputs.json`](sample_inputs.json)) used in the local Docker test was sent straight to the public HTTPS URL:

**Request**
```
POST https://p6si42spqz.us-east-1.awsapprunner.com/predict
Content-Type: application/json

{
  "customer_id": "C002",
  "review_text": "The top runs small and the stitching on one sleeve is already coming loose after a single wash. Disappointed for the price.",
  "age": 41,
  "positive_feedback_count": 1,
  "recommended_ind": 0,
  "division_name": "General Petite",
  "department_name": "Tops",
  "class_name": "Blouses",
  "image_available": true,
  "image": null
}
```

**Response** (`HTTP 200`)
```json
{
  "model_outputs": {
    "classical_score": 0.87,
    "cnn_score": 0.6,
    "rnn_score": 0.561,
    "final_score": 0.696,
    "risk_level": "medium"
  },
  "llm_explanation": {
    "summary": "Customer complaint details product quality issues (runs small, loose stitching after single wash) indicating dissatisfaction. The final model score suggests a moderate-to-high risk.",
    "recommended_action": "Escalate for human review to assess product quality concerns, investigate the complaint, and determine appropriate customer resolution.",
    "reason": "The final model score of 0.696 indicates a significant risk, and there is a strong disagreement among the individual model scores (Classical: 0.87 vs. RNN: 0.561), which necessitates human oversight for a comprehensive evaluation.",
    "human_review_required": true,
    "risk_notes": "The classical model identified a high risk (0.87), contrasting with the CNN (0.6) and RNN (0.561) models. The complaint specifically mentions product defects (sizing, stitching failure after first wash), which align with the higher end of the risk spectrum suggested by the classical model. This strong disagreement among model outputs triggers human review."
  },
  "_meta": { "cnn_mode": "simulated" }
}
```

`model_outputs` is bit-for-bit identical to the local run (both directly through [`hybrid_inference.py`](hybrid_inference.py) and through the local Docker container): the deterministic components (the classical model, the CNN, the RNN, the score combination, and the thresholding) behave identically in the cloud. Only the LLM's prose differs slightly between runs, which is expected, since it is generative and non-deterministic; the scores and the policy enforcement it is bound by do not vary, as discussed for the local runs in Q5 of [`evaluation_report.md`](evaluation_report.md).

## 6. Latency

Measured against the public URL from a residential connection, so these figures include real internet round-trip time, not just AWS-internal latency:

| Endpoint | Observed latency | Notes |
|---|---|---|
| `GET /` (health check) | **1.32 s** | No model inference; mostly network round trip plus ASGI overhead |
| `POST /predict` (run 1, includes Gemini's cold path) | **8.91 s** | |
| `POST /predict` (run 2) | **6.26 s** | |
| `POST /predict` (run 3) | **4.72 s** | |
| `POST /predict` (run 4) | **7.13 s** | |

`/predict` latency, roughly five to nine seconds, is dominated by the Gemini API round trip (the network hop from `us-east-1` to Google's API, plus generation time). This matches the roughly 3.5 to 5.9 second range observed for the same workload in the local Docker test, with the extra time here attributable to the additional public-internet hop between the client and App Runner. The three trained-model components (classical, CNN, and RNN) and the score combination run in well under a second; the LLM call is the bottleneck, exactly as the system's "LLM as explainer, not as sole decision-maker" design anticipates.

## 7. Gotchas encountered (useful for anyone repeating this)

1. **Architecture mismatch.** A plain `docker build` on the Apple Silicon build machine produces an `arm64` image, but App Runner expects `linux/amd64`. This was fixed with `docker buildx build --platform linux/amd64 ... --load`.
2. **`docker --env-file` does not strip quotes** the way `python-dotenv`'s `load_dotenv()` does. If `.env` stores `GEMINI_API_KEY='...'` with quotes, `--env-file` passes the literal quoted string, which causes a Gemini `400 API_KEY_INVALID` error, while `load_dotenv()` correctly strips them. This surfaced during the local Docker test, and was worked around by generating a quote-stripped temporary env file with `load_dotenv()` before calling `docker run --env-file`. This particular issue does not affect App Runner, since the key there comes from Secrets Manager as a raw string with no quoting involved.
3. **`describe-service` status lagged real readiness.** The CloudWatch `/aws/apprunner/.../application` log group showed `Uvicorn running on http://0.0.0.0:8000` and a `200 OK` health check response at `15:05:48 UTC`, and the public URL was already serving `200`s, yet `aws apprunner describe-service` kept reporting `OPERATION_IN_PROGRESS` for several more minutes before flipping to `RUNNING`. The lesson: when verifying an App Runner deployment, check the actual `ServiceUrl` directly (and the `application` log group) rather than relying solely on the `Status` field.

## 8. Cleanup (run after testing, to stop all charges)

After the live test above, and with the user's explicit confirmation (since these are account-modifying, cost-stopping operations on real AWS resources), the following were torn down in order, App Runner first, since that is where the compute billing comes from:

```bash
# 1. Delete the App Runner service (stops compute billing immediately)
aws apprunner delete-service --service-arn <service-arn>

# 2. Delete the ECR repository (and the image inside it)
aws ecr delete-repository --repository-name hybrid-risk-api --force

# 3. Delete the Secrets Manager secret (force, with no recovery window,
#    since it held nothing but a copy of a key that already lives in the local .env)
aws secretsmanager delete-secret \
  --secret-id hybrid-risk-api/GEMINI_API_KEY --force-delete-without-recovery

# 4. Detach policies and delete both IAM roles
aws iam detach-role-policy --role-name AppRunnerECRAccessRole-hybrid-risk \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
aws iam delete-role --role-name AppRunnerECRAccessRole-hybrid-risk

aws iam delete-role-policy --role-name AppRunnerInstanceRole-hybrid-risk --policy-name read-gemini-secret
aws iam delete-role --role-name AppRunnerInstanceRole-hybrid-risk
```

**Cleanup confirmation, verified live immediately after deletion:**

| Resource | Verification command | Result |
|---|---|---|
| App Runner service `hybrid-risk-api` | `describe-service` -> `Service.Status` | `DELETED` |
| ECR repository `hybrid-risk-api` | `describe-repositories --repository-names hybrid-risk-api` | `RepositoryNotFoundException: ... does not exist` |
| Secret `hybrid-risk-api/GEMINI_API_KEY` | `list-secrets` filtered by name | `[]` (empty) |
| IAM roles `AppRunner{ECRAccess,Instance}Role-hybrid-risk` | `list-roles` filtered by `hybrid-risk` | `[]` (empty) |

All five resources (the App Runner service, the ECR repository, the Secrets Manager secret, and both IAM roles) are confirmed deleted. No App Runner compute, ECR storage, or Secrets Manager resources for this project remain running or billable.
