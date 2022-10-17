# Kubeflow Pipelines Deployment

minikube start --name ml-in-production-course-week-4 --cpus='4' --memory='3g'

minikube start --namespace='ml-week-4' --cpus='4' --memory='3g'
kind create cluster --name ml-week4 --cpus=4 --memory=3g

minikube start --cpus='4' --memory='3g'

minikube kubectl create -f res.yaml
minikube kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
minikube kubectl create -f pipelines.yaml

export PIPELINE_VERSION=1.8.5
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" > kfp-yml/res.yaml
kubectl create -f kfp-yml/res.yaml

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION" > kfp-yml/pipelines.yaml
kubectl create -f kfp-yml/pipelines.yaml