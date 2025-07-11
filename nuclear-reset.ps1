# NUCLEAR OPTION - Full minikube reset
# Use this when everything else fails

Write-Host "=== NUCLEAR RESET ===" -ForegroundColor Red
Write-Host "This will completely reset minikube and all its data" -ForegroundColor Red
Write-Host "Press Ctrl+C now to abort, or wait 5 seconds to continue..." -ForegroundColor Yellow

# Wait for 5 seconds to allow cancellation
Start-Sleep -Seconds 5

# Kill all port-forwarding jobs
Get-Job | Where-Object { $_.Command -like '*port-forward*' } | Stop-Job
Get-Job | Where-Object { $_.Command -like '*port-forward*' } | Remove-Job

# Stop minikube
Write-Host "\nStopping minikube..." -ForegroundColor Yellow
minikube stop

# Delete minikube
Write-Host "\nDeleting minikube..." -ForegroundColor Yellow
minikube delete

# Start fresh minikube
Write-Host "\nStarting fresh minikube..." -ForegroundColor Green
minikube start `
  --driver=docker `
  --kubernetes-version=v1.30.1 `
  --cpus=8 --memory=12g `
  --wait=all --wait-timeout=8m

# Enable addons
Write-Host "\nEnabling addons..." -ForegroundColor Green
$addons = @("ingress", "default-storageclass", "storage-provisioner", "volumesnapshots")
foreach ($a in $addons) { minikube addons enable $a }

# Start tunnel
Write-Host "\nStarting minikube tunnel..." -ForegroundColor Green
Start-Job { minikube tunnel }

# Create namespace
Write-Host "\nCreating rag namespace..." -ForegroundColor Green
kubectl create namespace rag

Write-Host "\nNuclear reset complete!" -ForegroundColor Green
Write-Host "Run './deploy-fixed.ps1' to deploy Cassandra with fixed configuration" -ForegroundColor Cyan
