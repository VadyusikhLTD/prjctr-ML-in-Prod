# WEEK 5


## Streamlit serving for model
To install streamlit run

```python3 -m pip install streamlit```

To build streamlit app docker run command 


    make build_app_streamlit
	

To run streamlit app in docker run command


    make run_app_streamlit
	

## Fast API for model + tests + CI

To build FastApi app docker run command 


    make build_fast_api


To run FastApi app in docker run command


    make run_fast_api


To test FastApi app locally run command


    make test_fast_api

To test FastApi app in docker run command

    make test_fast_api_dockered

## Seldon API for model + tests


Clean all 

    kind delete clusters --all
    snap remove kubectl


Set up

    kind create cluster --name ml-in-production-course-week-5  --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
    kind get clusters
    snap install kubectl --channel=1.21/stable --classic
    
    kubectl apply -f k8s/ambassador-operator-crds.yaml
    kubectl apply -n ambassador -f k8s/ambassador-operator-kind.yaml
    kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

    kubectl create namespace seldon-system

    helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts \
        --set usageMetrics.enabled=true \
        --set ambassador.enabled=true \
        --namespace seldon-system
