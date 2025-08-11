minikube -p minikube docker-env | Invoke-Expression
docker build -t rag-ingest:latest -f ingest/Dockerfile .
kubectl delete job ingest-repos -n rag
helm upgrade rag-demo .\helm\ -n rag