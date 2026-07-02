<#
.SYNOPSIS
Deploys the AutoWarp Swarm Engine to Google Cloud Vertex AI Custom Training Jobs.

.DESCRIPTION
This PowerShell script automatically packages the workspace, builds the Docker image 
using Google Cloud Build, and triggers a massively parallel 96-core Compute Engine 
execution via Vertex AI.

.EXAMPLE
.\deploy_to_vertex.ps1 -ProjectID "my-gcp-project" -BucketName "autowarp-swarm-data"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectID,

    [Parameter(Mandatory=$true)]
    [string]$BucketName
)

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "🔥 AutoWarp Swarm Cloud Deployer (Vertex AI) 🔥" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Ensure GCP is authenticated
Write-Host "`n[1] Verifying Google Cloud Platform Authentication..."
gcloud config set project $ProjectID

# 2. Build and Push Docker image using Cloud Build
$ImageURI = "gcr.io/$ProjectID/autowarp-swarm:latest"
Write-Host "`n[2] Submitting infrastructure to Google Cloud Build..."
Write-Host "Pushing Image: $ImageURI" -ForegroundColor Yellow

# Navigate to root to ensure all context is sent
Set-Location -Path "$PSScriptRoot\.."

# We use the generic Dockerfile
gcloud builds submit --tag $ImageURI -f Dockerfile_Vertex .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error staging Docker container. Aborting." -ForegroundColor Red
    exit 1
}

# 3. Submit Custom Job to Vertex AI
Write-Host "`n[3] Spawning 96-core Swarm in Vertex AI..."
Write-Host "Allocating: n2-highcpu-96 instance" -ForegroundColor Yellow

# We pass the GCP Bucket name as an env variable inside the container
gcloud ai custom-jobs create `
    --region="us-central1" `
    --display-name="autowarp-swarm-run-$(Get-Date -UFormat %s)" `
    --worker-pool-spec="machine-type=n2-highcpu-96,replica-count=1,container-image-uri=$ImageURI" `
    --args="--env,GCP_BUCKET_NAME=$BucketName,--env,RUNNING_IN_VERTEX=true,--env,GEMINI_API_KEY=$env:GEMINI_API_KEY"

Write-Host "`n✅ Swarm Deployed to Google Vertex AI Successfully!" -ForegroundColor Green
Write-Host "You can monitor the AI agents running in live time via the Vertex AI Cloud Console."
Write-Host "Once a physical Subluminal/Warp solution is found, the charts will be sent to the bucket."
Write-Host "-> Use '$PSScriptRoot\pull_cloud_data.py' locally anytime to check and download results!" -ForegroundColor Cyan
