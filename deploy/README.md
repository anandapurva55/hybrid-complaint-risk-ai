# Deploying to AWS

One CloudFormation stack, [`ecs-fargate.yaml`](ecs-fargate.yaml):

```
client --HTTPS--> CloudFront (+ WAF rate limit per IP) --HTTP--> ALB (accepts only CloudFront's IP ranges)
                                                                   --> ECS Fargate (2 vCPU / 4 GB, autoscaling 1..3 on CPU)
                                                                          |-- Bedrock (Nova Lite) via the task role (that one model only)
                                                                          `-- API key from Secrets Manager (generated at deploy time)
```

Also created: CloudWatch log group (7-day retention) and alarms (5xx, unhealthy task, fallback-explanation share > 40%, human-review share > 50%; optional email via `AlarmEmail`). The app emits CloudWatch EMF metrics per prediction (no complaint text).

## Option A: deploy by hand

```bash
aws ecr create-repository --repository-name hybrid-risk-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker buildx build --platform linux/amd64 -t <acct>.dkr.ecr.<region>.amazonaws.com/hybrid-risk-api:v3 --push .

aws cloudformation deploy --stack-name hybrid-risk-api --template-file deploy/ecs-fargate.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageUri=<image uri> VpcId=<vpc-id> SubnetIds=<subnet-a>,<subnet-b> [AlarmEmail=you@example.com]

URL=$(aws cloudformation describe-stacks --stack-name hybrid-risk-api --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)
SEC=$(aws secretsmanager list-secrets --query 'SecretList[?contains(Name,`ApiKeySecret`)].ARN|[0]' --output text)
KEY=$(aws secretsmanager get-secret-value --secret-id $SEC --query SecretString --output text)
curl -X POST $URL/predict -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d @one_sample.json
```

## Option B (recommended): keyless deploy from GitHub Actions

**Status:** steps 1 and 2 are done for this account/repo (stack `hybrid-risk-github-oidc`, role `hybrid-risk-github-deploy`, repository variable `AWS_DEPLOY_ROLE_ARN`, GitHub environment `production` with the owner as required reviewer and deployments restricted to `main`). The workflow has not been run yet: it needs the code merged to `main`.

1. Once, from your own admin identity (not root keys):
   `aws cloudformation deploy --stack-name hybrid-risk-github-oidc --template-file deploy/github-oidc.yaml --capabilities CAPABILITY_NAMED_IAM`
   (pass `CreateOidcProvider=false ExistingOidcProviderArn=<arn>` if the account already has the GitHub OIDC provider). It creates a role that only this repository's `production` environment can assume and that can only manage the `hybrid-risk-api` stack.
2. In the GitHub repo: add the output as the variable `AWS_DEPLOY_ROLE_ARN`, create the `production` environment with required reviewers.
3. Run the **Deploy to AWS** workflow ([`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)): tests + `cfn-lint` gate it, then it builds the amd64 image, deploys the stack and smoke-tests the URL.

## Tear down
`aws cloudformation delete-stack --stack-name hybrid-risk-api` (the ECR repo and, for Option B, the OIDC stack are separate).

## Why a stack instead of direct CLI calls, and why not App Runner
1. **App Runner** refuses new services for this account: `Account ... can't create new App Runner services` (visible once no role is passed; the "not authorized to pass this role" message that appears otherwise is misleading). An AWS-side account restriction, so ECS is used.
2. **Root credentials cannot pass roles.** Direct `ecs register-task-definition` / `apprunner create-service` calls made with root access keys reject *any* role (`Role is not valid`), while the same call without a role works. CloudFormation passes the roles itself, so the stack deploys even from root; Option B avoids root altogether.

## Verified on the live deployment (2026-11-19)
HTTPS 200 on the CloudFront URL; plain HTTP redirects (301); the ALB's own address is unreachable directly; `/predict` without the key returns 401; a real prediction returns a decision plus a Bedrock-written explanation; EMF metrics appear in the log group; WAF blocked one IP after ~600 requests in about 2 minutes (the rate limit is evaluated with a short lag).

## Limitations
- `*.cloudfront.net` certificate only; a custom domain needs your own ACM certificate (us-east-1) and an alias on the distribution.
- The WAF rule limits per source IP only; the managed rule groups (SQLi, bots) are not enabled.
- Autoscaling is CPU-based; `/stats` stays per task, the EMF metrics are the fleet view.
- A running stack costs roughly $3-6/day (ALB, Fargate, CloudFront/WAF requests); delete it when idle.
