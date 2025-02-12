#!/bin/bash
set -e

# Variables
ECR_REPOSITORY="image-recognition-api"
AWS_REGION="ap-northeast-1"
IMAGE_TAG="latest"

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPOSITORY || \
aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --image-scanning-configuration scanOnPush=true \
    --region $AWS_REGION

# Get repository URI
REPOSITORY_URI=$(aws ecr describe-repositories \
    --repository-names $ECR_REPOSITORY \
    --query 'repositories[0].repositoryUri' \
    --output text)

# Build image
echo "Building Docker image..."
docker build -t $ECR_REPOSITORY:$IMAGE_TAG .

# Tag image
docker tag $ECR_REPOSITORY:$IMAGE_TAG $REPOSITORY_URI:$IMAGE_TAG

# Push image
echo "Pushing image to ECR..."
docker push $REPOSITORY_URI:$IMAGE_TAG

echo "Image successfully pushed to ECR: $REPOSITORY_URI:$IMAGE_TAG"
