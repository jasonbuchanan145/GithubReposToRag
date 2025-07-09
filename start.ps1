# Stage 1 – prep
# Stop on any non-handled error and enable strict mode
$ErrorActionPreference = 'Stop'

minikube start `
  --driver=docker `
  --kubernetes-version=v1.30.1 `
  --cpus=8 --memory=12g `
  --wait=all --wait-timeout=8m

# 3. Enable only the addons you need, after the API-server is healthy
$addons = @("ingress", "default-storageclass", "storage-provisioner", "volumesnapshots")
foreach ($a in $addons) { minikube addons enable $a }

# 4. (optional) run `minikube tunnel` in the background
Start-Job { minikube tunnel }

# Stage 2 – build images# --- Stage 2 – build images -----------------------------------------------
(minikube -p minikube docker-env) | Invoke-Expression
docker build -t rag-ingest:latest     -f scripts/Dockerfile              .
docker build -t rag-api:latest        -f services/rag_api/Dockerfile     .
docker build -t rag-frontend:latest   -f frontend/nextjs-app/Dockerfile  .
kubectl create namespace rag

# Stage 3 – deploy
helm upgrade --install rag-demo ./helm -n rag --create-namespace `
  --set image.tag=dev `
  --set image.pullPolicy=IfNotPresent


# Stage 4 – smoke-test
kubectl -n rag rollout status statefulset/cassandra --timeout=600s
kubectl -n rag create job --from=cronjob/ingest-repos ingest-manual

# Port-forward in background
Start-Job { kubectl -n rag port-forward svc/rag-api      8000:8000 }
Start-Job { kubectl -n rag port-forward svc/rag-frontend 3000:80   }

# Test query
#$body = @{ query = 'What does my ERC20 contract do?' } | ConvertTo-Json
#Invoke-RestMethod -Uri http://localhost:8000/rag -Method Post -ContentType 'application/json' -Body $body

#Start-Process http://localhost:3000