[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory)]
    [string]$GithubUser
)

# Stage 1 – prep
# Stop on any non-handled error and enable strict mode
$ErrorActionPreference = 'Stop'

# Handle GitHub user parameter
if (-not $GithubUser) {
    Write-Host ""
    Write-Host "GitHub User Configuration" -ForegroundColor Cyan
    Write-Host "Please specify the GitHub user whose repositories you want to ingest." -ForegroundColor White
    Write-Host "This can be your own username or any other GitHub user with public repositories." -ForegroundColor White
    Write-Host ""

    do {
        $GithubUser = Read-Host -Prompt "Enter GitHub username"
        if ([string]::IsNullOrWhiteSpace($GithubUser)) {
            Write-Host "Username cannot be empty. Please try again." -ForegroundColor Red
        }
    } while ([string]::IsNullOrWhiteSpace($GithubUser))
}

Write-Host "Using GitHub user: $GithubUser" -ForegroundColor Green

minikube start `
  --driver=docker `
  --container-runtime=docker --gpus=all --cpus=8 --memory=16g `
  --wait=all --wait-timeout=8m

# 3. Enable only the addons you need, after the API-server is healthy
$addons = @("ingress", "default-storageclass", "storage-provisioner", "volumesnapshots")
foreach ($a in $addons) { minikube addons enable $a }

# 4. (optional) run `minikube tunnel` in the background
Start-Job { minikube tunnel }

# Stage 2 – build images
# Re-eval docker-env as suggested by minikube
(minikube -p minikube docker-env) | Invoke-Expression



Write-Host "Building and loading Docker images into Minikube..." -ForegroundColor Yellow
try {
  # Switch to minikube's Docker environment
  Write-Host "Switching to minikube Docker environment..." -ForegroundColor Green
  & minikube -p minikube docker-env | Invoke-Expression

  Write-Host "Building rag-ingest..." -ForegroundColor Green
  docker build -t rag-ingest:latest -f ingest/Dockerfile .

  # Tag the image with no registry prefix for local use
  docker tag rag-ingest:latest rag-ingest:latest

  Write-Host "Building rag-api..." -ForegroundColor Green
  docker build -t localhost:5000/rag-api:latest -f rest_api/Dockerfile .
  docker tag localhost:5000/rag-api:latest rag-api:latest

  Write-Host "Building rag-worker..." -ForegroundColor Green

  docker build -t localhost:5000/rag-worker:latest -f rag_worker/Dockerfile .
  docker tag localhost:5000/rag-worker:latest rag-worker:latest

  Write-Host "All Docker images built successfully and loaded into Minikube!" -ForegroundColor Green
} catch {
  Write-Host "Docker build failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}

# Stage 3 – deploy

# Clean up any previous failed deployments
Write-Host "Cleaning up any previous failed deployments..."
try {
  # First, try to uninstall Helm release
  $helmReleases = helm list -n rag -q 2>$null
  if ($helmReleases -and $helmReleases.Contains("rag-demo")) {
    Write-Host "Uninstalling existing Helm release..."
    helm uninstall rag-demo -n rag --wait --timeout 120s
    Write-Host "Helm release uninstalled successfully"
  }
} catch {
  Write-Host "No previous Helm release to clean up or error during cleanup"
}

# Force cleanup of namespace if it exists and is stuck
$namespaceExists = $false
try {
  $namespaceCheck = kubectl get namespace rag -o name 2>$null
  if ($namespaceCheck) {
    $namespaceExists = $true
  }
} catch {
  # Namespace doesn't exist, which is fine
}

if ($namespaceExists) {
  Write-Host "Namespace 'rag' exists. Checking its status..."
  try {
    $namespaceStatus = kubectl get namespace rag -o jsonpath="{.status.phase}" 2>$null

    if ($namespaceStatus -eq "Terminating") {
      Write-Host "Namespace is stuck in Terminating state. Forcing cleanup..."

      # Try to remove finalizers from the namespace
      try {
        kubectl get namespace rag -o json | ConvertFrom-Json | ForEach-Object {
          $_.spec.finalizers = @()
          $_ | ConvertTo-Json -Depth 10 | kubectl replace --raw /api/v1/namespaces/rag/finalize -f -
        }
      } catch {
        Write-Host "Could not remove finalizers, continuing..."
      }

      # Wait for namespace to be fully deleted
      $timeout = 60
      $startTime = Get-Date
      while ($namespaceExists -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
        Write-Host "Waiting for namespace to be deleted..."
        Start-Sleep -Seconds 5
        try {
          $namespaceCheck = kubectl get namespace rag -o name 2>$null
          $namespaceExists = $namespaceCheck -ne $null
        } catch {
          $namespaceExists = $false
        }
      }

      if ($namespaceExists) {
        Write-Host "Namespace still exists after cleanup attempt. You may need to restart minikube."
        Write-Host "Run: minikube stop && minikube start"
        exit 1
      }
    } else {
      Write-Host "Deleting existing namespace..."
      kubectl delete namespace rag --timeout=60s

      # Wait for namespace to be fully deleted
      $timeout = 60
      $startTime = Get-Date
      while ($namespaceExists -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
        Write-Host "Waiting for namespace to be deleted..."
        Start-Sleep -Seconds 5
        try {
          $namespaceCheck = kubectl get namespace rag -o name 2>$null
          $namespaceExists = $namespaceCheck -ne $null
        } catch {
          $namespaceExists = $false
        }
      }
    }
  } catch {
    Write-Host "Error checking namespace status, proceeding with deletion..."
    try {
      kubectl delete namespace rag --timeout=60s 2>$null
    } catch {
      # Ignore errors if namespace doesn't exist
    }
  }
}

# Create fresh namespace
Write-Host "Creating fresh namespace..."
kubectl create namespace rag

# Check if GitHub token secret exists, if not prompt for it
Write-Host "Checking for GitHub token secret..." -ForegroundColor Yellow
$secretExists = $false
try {
  $secretCheck = kubectl -n rag get secret github-token -o name 2>$null
  if ($secretCheck) {
    $secretExists = $true
    Write-Host "GitHub token secret already exists" -ForegroundColor Green
  }
} catch {
  # Secret doesn't exist, which is expected for first run
}

if (-not $secretExists) {
  Write-Host ""
  Write-Host "GitHub Token Required" -ForegroundColor Cyan
  Write-Host "The ingestion service needs a GitHub personal access token to fetch repositories." -ForegroundColor White
  Write-Host "Please create a token at: https://github.com/settings/tokens" -ForegroundColor White
  Write-Host "Required scopes: repo (for private repos) or public_repo (for public repos only)" -ForegroundColor White
  Write-Host ""

  do {
    $githubToken = Read-Host -Prompt "Enter your GitHub personal access token" -AsSecureString
    $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($githubToken))

    if ([string]::IsNullOrWhiteSpace($plainToken)) {
      Write-Host "Token cannot be empty. Please try again." -ForegroundColor Red
    } elseif ($plainToken.Length -lt 20) {
      Write-Host "Token seems too short. GitHub tokens are typically 40+ characters. Please try again." -ForegroundColor Red
    }
  } while ([string]::IsNullOrWhiteSpace($plainToken) -or $plainToken.Length -lt 20)

  Write-Host "Creating GitHub token secret..." -ForegroundColor Green
  try {
    kubectl -n rag create secret generic github-token --from-literal=token=$plainToken
    Write-Host "GitHub token secret created successfully!" -ForegroundColor Green

    # Clear the token from memory for security
    $plainToken = $null
    $githubToken = $null
    [System.GC]::Collect()

  } catch {
    Write-Host "❌ Failed to create GitHub token secret: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You can create it manually with:" -ForegroundColor Yellow
    Write-Host "kubectl -n rag create secret generic github-token --from-literal=token=<your-token>" -ForegroundColor Yellow
    exit 1
  }
}

# Update dependencies and install Helm chart
Write-Host "Updating Helm dependencies..."
helm dependency update ./helm

Write-Host "Installing Helm chart with Cassandra..."
# For development environments, you can disable persistence if having PVC issues
$disablePersistence = $false # Set to $true to disable persistence for quick testing

try {
  if ($disablePersistence) {
    Write-Host "Running WITHOUT persistence - data will be lost when pods are deleted" -ForegroundColor Yellow
    helm install rag-demo ./helm -n rag `
          --set image.tag=dev `
          --set image.pullPolicy=IfNotPresent `
          --set cassandra.persistence.enabled=false `
          --set github.user=$GithubUser
  } else {
    helm install rag-demo ./helm -n rag `
          --set image.tag=dev `
          --set image.pullPolicy=IfNotPresent `
          --set github.user=$GithubUser
  }
  Write-Host "Helm chart installed successfully!" -ForegroundColor Green
} catch {
  Write-Host "Helm install failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}

# Wait for Cassandra to be ready
Write-Host "Waiting for Cassandra PVC to be bound..."
$pvcBound = $false
$timeout = 120
$startTime = Get-Date

while (-not $pvcBound -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
  try {
    $pvcs = kubectl -n rag get pvc -l app.kubernetes.io/name=cassandra -o jsonpath="{.items}" 2>$null
    if ($pvcs -and $pvcs -ne "[]") {
      $pvcStatus = kubectl -n rag get pvc -l app.kubernetes.io/name=cassandra -o jsonpath="{.items[0].status.phase}" 2>$null
      if ($pvcStatus -eq "Bound") {
        $pvcBound = $true
        Write-Host "PVC successfully bound!" -ForegroundColor Green
      } else {
        Write-Host "PVC Status: $pvcStatus - waiting for PVC to be bound..."
        kubectl -n rag get pvc 2>$null
      }
    } else {
      Write-Host "No PVCs found yet, waiting..."
    }
  } catch {
    Write-Host "Waiting for PVCs to be created..."
  }

  if (-not $pvcBound) {
    Start-Sleep -Seconds 5
  }
}

if (-not $pvcBound) {
  Write-Host "PVC did not bind within timeout. Continuing anyway..." -ForegroundColor Yellow
}

Write-Host "Waiting for Cassandra pod to be created..."
$cassandraPod = ""
$timeout = 60
$startTime = Get-Date

while (-not $cassandraPod -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
  try {
    $pods = kubectl -n rag get pods -l app.kubernetes.io/name=cassandra -o jsonpath="{.items}" 2>$null
    if ($pods -and $pods -ne "[]") {
      $cassandraPod = kubectl -n rag get pods -l app.kubernetes.io/name=cassandra -o jsonpath="{.items[0].metadata.name}" 2>$null
      if ($cassandraPod) {
        Write-Host "Found Cassandra pod: $cassandraPod"
      }
    } else {
      Write-Host "No Cassandra pods found yet, waiting..."
    }
  } catch {
    Write-Host "Waiting for Cassandra pods to be created..."
  }

  if (-not $cassandraPod) {
    Start-Sleep -Seconds 5
  }
}

if (-not $cassandraPod) {
  Write-Host "Cassandra pod was not created within timeout" -ForegroundColor Red
  Write-Host "Checking deployment status..." -ForegroundColor Yellow
  kubectl -n rag get all
  exit 1
}

Write-Host "Monitoring Cassandra pod $cassandraPod startup..."
$timeout = 300
$startTime = Get-Date
$ready = $false

while (-not $ready -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout) {
  try {
    $podStatus = kubectl -n rag get pod $cassandraPod -o jsonpath="{.status.phase}" 2>$null
    Write-Host "Current pod status: $podStatus"

    # Show the latest logs
    Write-Host "Latest Cassandra logs:"
    kubectl -n rag logs $cassandraPod --tail=10 2>$null

    # Check if pod is ready
    $readyStatus = kubectl -n rag get pod $cassandraPod -o jsonpath="{.status.containerStatuses[0].ready}" 2>$null
    if ($readyStatus -eq "true") {
      $ready = $true
      Write-Host "Cassandra pod is ready!" -ForegroundColor Green
    } else {
      Write-Host "Waiting for Cassandra pod to be ready... (will retry in 15 seconds)"
      Start-Sleep -Seconds 15
    }
  } catch {
    Write-Host "Error checking pod status, will retry..."
    Start-Sleep -Seconds 15
  }
}

if (-not $ready) {
  Write-Host "Cassandra pod did not become ready within timeout period" -ForegroundColor Red
  Write-Host "Checking pod events for more information:" -ForegroundColor Yellow
  kubectl -n rag describe pod $cassandraPod
  Write-Host "You can continue, but services may not work correctly." -ForegroundColor Yellow
} else {
  Write-Host "Cassandra is ready!" -ForegroundColor Green
}

Write-Host "Deployment completed! In order to access from your host machine please run `minikube services --all -n rag" -ForegroundColor Green