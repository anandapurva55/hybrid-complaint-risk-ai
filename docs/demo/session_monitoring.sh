# CloudWatch alarms and the app's own metrics, plus CloudFront and WAF. Needs START (ISO time) in the environment.
source "$(dirname "$0")/lib.sh"
type_cmd "aws cloudwatch describe-alarms --alarm-name-prefix hybrid-risk-api --query 'MetricAlarms[].{alarm:AlarmName,state:StateValue}' --output table" 1
type_cmd "aws cloudwatch get-metric-statistics --namespace HybridRiskApi --metric-name Predictions --dimensions Name=Service,Value=hybrid-risk-api --start-time $START --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) --period 300 --statistics Sum --query 'sort_by(Datapoints,&Timestamp)[].{five_minutes_from:Timestamp,predictions:Sum}' --output table" 1
type_cmd "aws cloudfront list-distributions --query 'DistributionList.Items[].{domain:DomainName,status:Status,enabled:Enabled}' --output table" 1
type_cmd "aws wafv2 list-web-acls --scope CLOUDFRONT --region us-east-1 --query 'WebACLs[].Name' --output text" 6
