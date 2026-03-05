# LegalMind — Kubernetes Deployment Guide

## 🎯 Overview

This guide covers deploying LegalMind to Google Kubernetes Engine (GKE) with:

- **Namespace isolation**
- **ConfigMaps** for configuration
- **Secrets** for API keys
- **Deployment** with health probes
- **HPA** (Horizontal Pod Autoscaler)
- **LoadBalancer Service**

---

## 🚀 Quick Start

### Option 1: Automated Deployment

```bash
# Set your GCP project
export GCP_PROJECT_ID=your-project-id

# Run deployment script
./deploy-gke.sh
```

### Option 2: Manual Deployment

#### Step 1: Create GKE Cluster

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create cluster
gcloud container clusters create legalmind-cluster \
    --zone=me-central1-a \
    --num-nodes=2 \
    --machine-type=e2-standard-2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5

# Get credentials
gcloud container clusters get-credentials legalmind-cluster --zone=me-central1-a
```

#### Step 2: Build and Push Docker Image

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/legalmind:latest .

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/legalmind:latest
```

#### Step 3: Create Kubernetes Resources

```bash
# Create namespace
kubectl apply -f k8s/00-namespace.yaml

# Create secrets (replace with your actual keys)
kubectl create secret generic legalmind-secrets \
    --namespace=legalmind \
    --from-literal=OPENAI_API_KEY=sk-your-key \
    --from-literal=COHERE_API_KEY=your-cohere-key

# Apply remaining manifests
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/03-deployment.yaml
kubectl apply -f k8s/04-service.yaml
kubectl apply -f k8s/05-hpa.yaml
```

#### Step 4: Get External IP

```bash
# Wait for LoadBalancer IP
kubectl get svc legalmind-lb -n legalmind -w

# Access the application
curl http://EXTERNAL_IP/health
```

---

## ☸️ Kubernetes Concepts Explained

### 1. Namespace (`00-namespace.yaml`)

- **What**: Logical isolation boundary for resources
- **Why**: Separates LegalMind from other workloads
- **Key**: All resources use `namespace: legalmind`

### 2. ConfigMap (`01-configmap.yaml`)

- **What**: Non-sensitive configuration stored as key-value pairs
- **Why**: Separates config from code, easy to update without rebuilding
- **Key**: Mounted as environment variables in pods

### 3. Secret (`02-secret.yaml`)

- **What**: Sensitive data (API keys) stored encrypted
- **Why**: Security — secrets are base64 encoded at rest
- **Key**: Referenced via `secretKeyRef` in deployment

### 4. Deployment (`03-deployment.yaml`)

- **What**: Declarative pod management with replicas
- **Why**: Ensures desired state, handles rollouts/rollbacks
- **Key Components**:
  - `replicas: 2` — Run 2 instances for HA
  - `resources` — CPU/memory limits prevent noisy neighbors
  - `livenessProbe` — Restarts unhealthy containers
  - `readinessProbe` — Only routes traffic when ready
  - `strategy: RollingUpdate` — Zero-downtime deployments

### 5. Service (`04-service.yaml`)

- **What**: Stable network endpoint for pods
- **Types**:
  - `ClusterIP` — Internal only
  - `LoadBalancer` — External access via cloud LB
- **Key**: Selector matches deployment labels

### 6. HPA (`05-hpa.yaml`)

- **What**: Automatically scales pods based on metrics
- **Why**: Handle variable load efficiently
- **Key**:
  - `minReplicas: 2` — Always maintain HA
  - `maxReplicas: 10` — Cap for cost control
  - `averageUtilization: 70` — Scale when CPU > 70%

---

## 📊 Useful Commands

```bash
# View pods
kubectl get pods -n legalmind

# View pod logs
kubectl logs -f deployment/legalmind-api -n legalmind

# View HPA status
kubectl get hpa -n legalmind

# Describe deployment (troubleshooting)
kubectl describe deployment legalmind-api -n legalmind

# Scale manually
kubectl scale deployment legalmind-api -n legalmind --replicas=3

# View all resources
kubectl get all -n legalmind

# Port forward for local testing
kubectl port-forward svc/legalmind-service -n legalmind 8080:80
```

---

## 🔧 Troubleshooting

### Pods not starting?

```bash
kubectl describe pod <pod-name> -n legalmind
kubectl logs <pod-name> -n legalmind
```

### API key issues?

```bash
# Verify secrets exist
kubectl get secrets -n legalmind

# View secret (encoded)
kubectl get secret legalmind-secrets -n legalmind -o yaml
```

### Image pull errors?

```bash
# Check image exists in GCR
gcloud container images list --repository=gcr.io/YOUR_PROJECT_ID
```

---

## 💰 Cost Optimization

To minimize costs during demo:

```bash
# Scale down when not in use
kubectl scale deployment legalmind-api -n legalmind --replicas=0

# Delete cluster when done
gcloud container clusters delete legalmind-cluster --zone=me-central1-a
```