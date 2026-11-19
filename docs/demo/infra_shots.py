"""
Renders terminal screenshots of REAL AWS CLI output from a running deployment of the stack, plus a pytest run:

  docs/images/infra_ecs.png         ECS service (running task) and ALB target health
  docs/images/infra_monitoring.png  CloudWatch alarms and the app's own metrics, CloudFront distribution, WAF
  docs/images/tests.png             a real `pytest -q` run

    python docs/demo/infra_shots.py [--stack hybrid-risk-api]

Run while the stack is up. 12-digit AWS account IDs are redacted by the renderer; no secret is ever printed
(the API key is not used here). Needs AWS credentials and Pillow.
"""
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from term import DIM, FG, GREEN, Terminal, wrap_command  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images"
STACK = sys.argv[sys.argv.index("--stack") + 1] if "--stack" in sys.argv else "hybrid-risk-api"
TITLE_TAIL = f"captured {date.today()} from the live deployment (since torn down)"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.rstrip("\n")


def physical(logical):
    return sh(f"aws cloudformation describe-stack-resource --stack-name {STACK} --logical-resource-id {logical} "
              "--query StackResourceDetail.PhysicalResourceId --output text")


def shot(name, title, steps, cols=132, tail=TITLE_TAIL):
    """steps: list of (displayed_command, real_command_or_None). Draws every command and its output."""
    lines = []
    for shown, real in steps:
        for i, t in enumerate(wrap_command(shown, cols)):
            lines.append([("$ " if i == 0 else "  ", GREEN), (t, FG)])
        out = sh(real or shown)
        lines += [[(l, FG)] for l in out.split("\n")] if out else []
        lines.append([("", FG)])
    term = Terminal(f"{title}: {tail}", cols=cols, rows=len(lines) + 1)
    term.frame(lines).save(IMAGES / name)
    print("wrote", name, term.W, "x", term.H)


def main():
    cluster = physical("Cluster")
    service = sh(f"aws ecs list-services --cluster {cluster} --query 'serviceArns[0]' --output text").split("/")[-1]
    tg = physical("TargetGroup")
    shot("infra_ecs.png", "aws cli: ECS on Fargate", [
        (f"aws ecs describe-services --cluster {cluster} --services {service} --query 'services[0].{{service:serviceName,status:status,desired:desiredCount,running:runningCount,rollout:deployments[0].rolloutState,launchType:launchType}}'", None),
        (f"aws ecs describe-tasks --cluster {cluster} --tasks $(aws ecs list-tasks --cluster {cluster} --query 'taskArns[0]' --output text) --query 'tasks[0].{{status:lastStatus,health:healthStatus,cpu:cpu,memory:memory,platform:platformVersion}}'",
         f"aws ecs describe-tasks --cluster {cluster} --tasks $(aws ecs list-tasks --cluster {cluster} --query 'taskArns[0]' --output text) --query 'tasks[0].{{status:lastStatus,health:healthStatus,cpu:cpu,memory:memory,platform:platformVersion}}'"),
        (f"aws elbv2 describe-target-health --target-group-arn {tg} --query 'TargetHealthDescriptions[].{{target:Target.Id,port:Target.Port,state:TargetHealth.State}}'", None),
    ])
    now = datetime.now(timezone.utc)
    # window the metric to THIS deployment: the metric name is shared by every deployment of the stack
    created = sh(f"aws cloudformation describe-stacks --stack-name {STACK} --query 'Stacks[0].CreationTime' --output text")
    start = (datetime.fromisoformat(created).astimezone(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    shot("infra_monitoring.png", "aws cli: CloudWatch, CloudFront, WAF", [
        (f"aws cloudwatch describe-alarms --alarm-name-prefix {STACK} --query 'MetricAlarms[].{{alarm:AlarmName,state:StateValue}}' --output table", None),
        (f"aws cloudwatch get-metric-statistics --namespace HybridRiskApi --metric-name Predictions --dimensions Name=Service,Value=hybrid-risk-api --start-time {start} --end-time {end} --period 300 --statistics Sum --query 'sort_by(Datapoints,&Timestamp)[].{{five_minutes_from:Timestamp,predictions:Sum}}' --output table", None),
        ("aws cloudfront list-distributions --query 'DistributionList.Items[].{domain:DomainName,status:Status,enabled:Enabled}' --output table", None),
        ("aws wafv2 list-web-acls --scope CLOUDFRONT --region us-east-1 --query 'WebACLs[].Name' --output text", None),
    ])
    shot("tests.png", "pytest", [("python -m pytest -q", f"{sys.executable} -m pytest -q 2>&1 | tail -6")], cols=90, tail=f"local run, {date.today()}")


if __name__ == "__main__":
    main()
