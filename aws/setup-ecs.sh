#!/bin/bash
set -e

# Variables
CLUSTER_NAME="image-recognition-cluster"
SERVICE_NAME="image-recognition-api"
TASK_FAMILY="image-recognition-api"
DESIRED_COUNT=2

# Create ECS cluster
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --tags key=Environment,value=Production \
    --capacity-providers FARGATE

# Create security group
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name "image-recognition-api-sg" \
    --description "Security group for image recognition API" \
    --query 'GroupId' \
    --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count $DESIRED_COUNT \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --tags key=Environment,value=Production

# Create auto scaling
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --min-capacity 2 \
    --max-capacity 10

# Create scaling policies
aws application-autoscaling put-scaling-policy \
    --policy-name cpu-tracking-scaling \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        }
    }'

# Create CloudWatch log group
aws logs create-log-group \
    --log-group-name /ecs/image-recognition-api

# Set log retention
aws logs put-retention-policy \
    --log-group-name /ecs/image-recognition-api \
    --retention-in-days 30
