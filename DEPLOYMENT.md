# Deployment Manual: B2B PaddleOCR Service on Google Cloud Run

This document details the exact end-to-end instructions for deploying the `b2b-ocr` microservice to Google Cloud Run using Google Artifact Registry and Google Cloud Build.

---

## 1. Google Cloud Architecture

```
[ B2B Express Gateway ] ─── (IAM Authenticated HTTP Request) ───► [ Google Cloud Run: b2b-ocr ]
                                                                       │
                                                                       ├── 2 vCPUs / 2Gi RAM
                                                                       ├── Auto-scaling (0-10)
                                                                       └── Port 8000 ($PORT)
```

---

## 2. Environment Variables & Placeholders

| Variable | Description | Example |
| :--- | :--- | :--- |
| `PROJECT_ID` | Your Google Cloud Project ID | `zain-b2b-prod` |
| `REGION` | Compute Region | `europe-west3` |
| `REPO_NAME` | Artifact Registry Docker Repo | `b2b-ocr-repo` |
| `SERVICE_NAME` | Cloud Run Service Name | `b2b-ocr` |

---

## 3. Step-by-Step Deployment

### Step A: Authenticate & Enable Required APIs
```bash
gcloud auth login
gcloud config set project PROJECT_ID

gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com
```

### Step B: Create Artifact Registry Repository
```bash
gcloud artifacts repositories create REPO_NAME \
    --repository-format=docker \
    --location=REGION \
    --description="Docker repository for B2B PaddleOCR Service"
```

### Step C: Build & Push Image via Cloud Build
```bash
# Run from within the b2b-ocr repository root
gcloud builds submit \
    --tag REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/b2b-ocr:latest .
```

### Step D: Deploy to Google Cloud Run
```bash
gcloud run deploy SERVICE_NAME \
    --image=REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/b2b-ocr:latest \
    --platform=managed \
    --region=REGION \
    --port=8000 \
    --cpu=2 \
    --memory=2Gi \
    --min-instances=0 \
    --max-instances=10 \
    --concurrency=8 \
    --timeout=60s \
    --no-allow-unauthenticated
```

---

## 4. Service-to-Service Security (IAM)

To allow the main B2B backend service account to securely invoke the OCR service:

```bash
# Grant Cloud Run Invoker role to the Main B2B backend service account
gcloud run services add-iam-policy-binding SERVICE_NAME \
    --region=REGION \
    --member="serviceAccount:MAIN_B2B_SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

---

## 5. Connecting with Main B2B Backend

After deployment, copy the generated Cloud Run URL (`https://b2b-ocr-xyz.a.run.app`) and configure it in the main B2B Node.js backend environment:

```bash
PADDLE_OCR_SERVICE_URL="https://b2b-ocr-xyz.a.run.app"
```

---

## 6. Verification & Health Monitoring

### Health Check:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     https://b2b-ocr-xyz.a.run.app/health
```

### Cloud Logging:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=b2b-ocr" --limit=50
```
