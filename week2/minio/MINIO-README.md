# MINIO deployment 

Firstly I've tried to deploy MINIO using their [official deployment guide](https://github.com/kubernetes/examples/tree/master/staging/storage/minio), 
but unfortunately I've failed to deploy it. So I used [yours deployment guide](!https://github.com/truskovskiyk/ml-in-production-webinars/tree/main/week-2).
My problem was - I've _minio-service_ instead of _minio-api_ and _minio-ui_.

## Deployment 
1. Install _kontena-lens_ as UI to observe kubernetes.


    kontena-lens
2. Delete previous and start new _minikube_


    sudo minikube delete
    minikube start

3. Start _minio_ 


    kubectl create -f minio-standalone.yaml

4. Enable ports to UI and API 

    
    kubectl port-forward svc/minio-ui 9001:9001
    kubectl port-forward svc/minio-api 9000:9000

