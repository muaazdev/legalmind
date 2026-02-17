#!/bin/bash
# ============================================
# LegalMind GKE Deployment Script
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 LegalMind GKE Deployment Script${NC}"
echo "============================================"

# ============================================
# Configuration - EDIT THESE VALUES
# ============================================
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="me-central1"  
ZONE="${REGION}-a"
CLUSTER_NAME="legalmind-cluster"
IMAGE_NAME="gcr.io/${PROJECT_ID}/legalmind"
IMAGE_TAG="latest"

# ============================================
# Pre-flight checks
# ============================================
echo -e "\n${YELLOW}📋 Pre-flight checks...${NC}"

# Check gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Install it from https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Check kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Installing...${NC}"
    gcloud components install kubectl
fi

# Check docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker not found. Please install Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All tools available${NC}"

# ============================================
# Step 1: Authenticate and set project
# ============================================
echo -e "\n${YELLOW}🔐 Step 1: Setting up GCP project...${NC}"

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

echo -e "${GREEN}✅ Project set to ${PROJECT_ID}${NC}"

# ============================================
# Step 2: Enable required APIs
# ============================================
echo -e "\n${YELLOW}🔧 Step 2: Enabling required APIs...${NC}"

gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com

echo -e "${GREEN}✅ APIs enabled${NC}"

# ============================================
# Step 3: Create GKE Cluster (if not exists)
# ============================================
echo -e "\n${YELLOW}☸️ Step 3: Creating GKE cluster...${NC}"

if gcloud container clusters describe ${CLUSTER_NAME} --zone=${ZONE} &> /dev/null; then
    echo -e "${YELLOW}⚠️ Cluster ${CLUSTER_NAME} already exists${NC}"
else
    echo "Creating cluster (this takes 5-10 minutes)..."
    gcloud container clusters create ${CLUSTER_NAME} \
        --zone=${ZONE} \
        --num-nodes=2 \
        --machine-type=e2-standard-2 \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=5 \
        --enable-autorepair \
        --enable-autoupgrade
    
    echo -e "${GREEN}✅ Cluster created${NC}"
fi

# Get cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE}
echo -e "${GREEN}✅ kubectl configured for cluster${NC}"

# ============================================
# Step 4: Build and push Docker image
# ============================================
echo -e "\n${YELLOW}🐳 Step 4: Building and pushing Docker image...${NC}"

# Configure docker for GCR
gcloud auth configure-docker --quiet

# Build image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Push to GCR
docker push ${IMAGE_NAME}:${IMAGE_TAG}

echo -e "${GREEN}✅ Image pushed to ${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# ============================================
# Step 5: Update deployment manifest with project ID
# ============================================
echo -e "\n${YELLOW}📝 Step 5: Updating Kubernetes manifests...${NC}"

# Replace PROJECT_ID in deployment
sed -i "s|gcr.io/PROJECT_ID/legalmind|${IMAGE_NAME}|g" k8s/03-deployment.yaml

echo -e "${GREEN}✅ Manifests updated${NC}"

# ============================================
# Step 6: Create secrets (interactive)
# ============================================
echo -e "\n${YELLOW}🔑 Step 6: Creating Kubernetes secrets...${NC}"

# Create namespace first
kubectl apply -f k8s/00-namespace.yaml

# Check if secrets exist
if kubectl get secret legalmind-secrets -n legalmind &> /dev/null; then
    echo -e "${YELLOW}⚠️ Secrets already exist. Skipping...${NC}"
else
    echo "Enter your OpenAI API key:"
    read -s OPENAI_KEY
    echo "Enter your Cohere API key (press Enter to skip):"
    read -s COHERE_KEY
    
    if [ -z "$COHERE_KEY" ]; then
        COHERE_KEY="placeholder"
    fi
    
    kubectl create secret generic legalmind-secrets \
        --namespace=legalmind \
        --from-literal=OPENAI_API_KEY=${OPENAI_KEY} \
        --from-literal=COHERE_API_KEY=${COHERE_KEY}
    
    echo -e "${GREEN}✅ Secrets created${NC}"
fi

# ============================================
# Step 7: Deploy all manifests
# ============================================
echo -e "\n${YELLOW}🚀 Step 7: Deploying to Kubernetes...${NC}"

kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/03-deployment.yaml
kubectl apply -f k8s/04-service.yaml
kubectl apply -f k8s/05-hpa.yaml
kubectl apply -f k8s/06-ingress.yaml

echo -e "${GREEN}✅ All manifests applied${NC}"

# ============================================
# Step 8: Wait for deployment
# ============================================
echo -e "\n${YELLOW}⏳ Step 8: Waiting for deployment to be ready...${NC}"

kubectl rollout status deployment/legalmind-api -n legalmind --timeout=300s

echo -e "${GREEN}✅ Deployment ready${NC}"

# ============================================
# Step 9: Get external IP
# ============================================
echo -e "\n${YELLOW}🌐 Step 9: Getting external IP...${NC}"

echo "Waiting for LoadBalancer IP (this may take 1-2 minutes)..."
sleep 30

EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
    EXTERNAL_IP=$(kubectl get svc legalmind-lb -n legalmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$EXTERNAL_IP" ]; then
        echo "Still waiting for IP..."
        sleep 10
    fi
done

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "\nAccess LegalMind at: ${YELLOW}http://${EXTERNAL_IP}${NC}"
echo -e "\nUseful commands:"
echo -e "  kubectl get pods -n legalmind"
echo -e "  kubectl logs -f deployment/legalmind-api -n legalmind"
echo -e "  kubectl get hpa -n legalmind"
echo -e "\nTo delete the deployment:"
echo -e "  kubectl delete namespace legalmind"
echo -e "  gcloud container clusters delete ${CLUSTER_NAME} --zone=${ZONE}"
