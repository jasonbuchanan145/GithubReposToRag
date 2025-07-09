# HelloLLM
An LLM configured to provide assistance with all my projects


### Create helm secret

Because this project pulls from github you will need to supply your api key.
To do this after you have minikube turned on run 

```shell
kubectl -n rag create secret generic github-token \
  --from-literal=token=_insert_token_here_
```