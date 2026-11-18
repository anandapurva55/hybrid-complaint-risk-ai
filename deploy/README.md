# Deploying to AWS (ECS Fargate + ALB, Bedrock explanations)

[`ecs-fargate.yaml`](ecs-fargate.yaml) is a CloudFormation stack: ECS cluster and Fargate service (2 vCPU / 4 GB), an Application Load Balancer, least-privilege roles (the task role may invoke only the one Bedrock model), a 7-day log group, and an API key **generated in Secrets Manager at deploy time**.

```bash
# 1. Push the image (amd64) to ECR
aws ecr create-repository --repository-name hybrid-risk-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker buildx build --platform linux/amd64 -t <acct>.dkr.ecr.<region>.amazonaws.com/hybrid-risk-api:v2 --push .

# 2. Deploy the stack (default VPC shown; use two public subnets in different AZs)
aws cloudformation deploy --stack-name hybrid-risk-api --template-file deploy/ecs-fargate.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageUri=<image uri> VpcId=<vpc-id> SubnetIds=<subnet-a>,<subnet-b>

# 3. Use it
URL=$(aws cloudformation describe-stacks --stack-name hybrid-risk-api --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)
KEY=$(aws secretsmanager get-secret-value --secret-id <ApiKeySecretArn output> --query SecretString --output text)
curl -X POST $URL/predict -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d @one_sample.json

# 4. Tear down (removes everything the stack created)
aws cloudformation delete-stack --stack-name hybrid-risk-api
```

Requirements: Bedrock access to `amazon.nova-lite-v1:0` (via the `us.` inference profile) in the region.

## Why not App Runner, and why not direct `aws ecs`/`apprunner` calls

Two separate problems were found when redeploying this account:

1. **App Runner** refuses new services for this account: `Account ... can't create new App Runner services` (visible once no role is passed; the "not authorized to pass this role" message that appears otherwise is misleading). This is an AWS-side account restriction, so the stack uses ECS instead.
2. **Role passing from the root user fails.** Direct `ecs register-task-definition` / `apprunner create-service` calls made with root access keys reject *any* role (`Role is not valid`), while the same call without a role works. CloudFormation passes the roles itself, so deploying through the stack works even from root credentials. Better practice (and what the sibling claims project uses): deploy from an IAM role or SSO identity, e.g. GitHub Actions with OIDC, and avoid root access keys altogether.

## Limitations
- HTTP only (the ALB has no certificate). Put it behind CloudFront/ACM or add an HTTPS listener with your own domain before sending real data; the API key travels in a header.
- Single task, no autoscaling; `/stats` monitoring is per task.
- A running stack costs roughly $1.5-3/day (ALB plus 2 vCPU/4 GB Fargate); delete it when idle.
