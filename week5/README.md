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

### Set up

    kind create cluster --name ml-in-production-course-week-5  --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
    
    kubectl apply -f k8s/ambassador-operator-crds.yaml
    kubectl apply -n ambassador -f k8s/ambassador-operator-kind.yaml`
    kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

    kubectl create namespace seldon-system

    helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts \
        --set usageMetrics.enabled=true \
        --set ambassador.enabled=true \
        
    kubectl port-forward  --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
        --namespace seldon-system


#### Run standart iris model
Create seldon server with model 

    kubectl create -f k8s/seldon-iris.yaml
    
Run request

    curl -X POST "http://0.0.0.0:7777/seldon/default/iris-model/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"ndarray\":[[1,2,3,4]]}}"

Example of response 

    {"data":{"names":["t:0","t:1","t:2"],"ndarray":[[0.0006985194531162835,0.00366803903943666,0.995633441507447]]},"meta":{"requestPath":{"classifier":"seldonio/sklearnserver:1.14.1"}}}


#### Run custom fashion-mnist-classifier model
Create seldon server with model 

    kubectl create -f k8s/seldon-custom.yaml
    
Link 

    http://localhost:7777/seldon/default/fashion-mnist-classifier/api/v1.0/doc/#


### Troubleshooting 

Check or delete runnig clusters 

    kind get clusters
    kind delete clusters --all

Reinstall kubectl of version that works with seldon and ambassador 

    snap remove kubectl
    snap install kubectl --channel=1.21/stable --classic
