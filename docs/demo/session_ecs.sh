# The running ECS service, its task and the load-balancer target. Needs CLUSTER and SERVICE in the environment.
source "$(dirname "$0")/lib.sh"
type_cmd "aws ecs describe-services --cluster $CLUSTER --services $SERVICE --query 'services[0].{service:serviceName,status:status,desired:desiredCount,running:runningCount,rollout:deployments[0].rolloutState,launchType:launchType}'" 1
type_cmd "aws ecs describe-tasks --cluster $CLUSTER --tasks \$(aws ecs list-tasks --cluster $CLUSTER --query 'taskArns[0]' --output text) --query 'tasks[0].{status:lastStatus,health:healthStatus,cpu:cpu,memory:memory}'" 1
type_cmd "aws elbv2 describe-target-health --target-group-arn \$(aws elbv2 describe-target-groups --query 'TargetGroups[0].TargetGroupArn' --output text) --query 'TargetHealthDescriptions[].{target:Target.Id,port:Target.Port,state:TargetHealth.State}'" 6
