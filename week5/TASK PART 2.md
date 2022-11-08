# Optimization of ML models

## Презентація:

[https://docs.google.com/presentation/d/1WMS4-aG3H5RmEVNIifgUcREN6HVLHlGM/edit](https://docs.google.com/presentation/d/1WMS4-aG3H5RmEVNIifgUcREN6HVLHlGM/edit)

## **Домашнє завдання:**

### Materials:

- [Most Effective Types of Performance Testing](https://loadninja.com/articles/performance-test-types/)
- [SageMaker endpoints](https://github.com/awsdocs/amazon-sagemaker-developer-guide/blob/master/doc_source/realtime-endpoints-deployment.md)
- [Deploy your side-projects at scale for basically nothing - Google Cloud Run](https://alexolivier.me/posts/deploy-container-stateless-cheap-google-cloud-run-serverless)
- [Load testing tool vegeta](https://github.com/tsenart/vegeta)
- [Locust for load testing](https://github.com/locustio/locust)
- [The Top 174 Model Compression Open Source Projects](https://awesomeopensource.com/projects/model-compression)
- [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282)
- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Autoscaling Seldon Deployments](https://docs.seldon.io/projects/seldon-core/en/latest/examples/autoscaling_example.html)

### Task:

- PR benchmark for your model API
- PR benchmark for your model forward pass
- PR k8s deployment YAML for your model API:
    - Deployment YAML
    - Service YAML
- PR k8s deployment YAML for your model UI (Streamlit):
    - Deployment YAML
    - Service YAML
- PR optimizing inference for your model
    - Benchmark model as is
    - Optimize
    - Benchmark model after
- Google doc update about model benchmarking & optimization
- PR autoscaling for pod (optioal)
- PR GCP cloud run deployment (optional)
- PR GCP cloud run benchmarking (optional)
- PR aws fargate deployment (optional)
- PR aws fargate benchmarking (optional)

### Criteria:

- 5 PRs merged
- Doc is updated

## Додаткові матеріали:

### API:

- [https://github.com/SeldonIO/MLServer](https://github.com/SeldonIO/MLServer)
- [https://github.com/cortexlabs/nucleus](https://github.com/cortexlabs/nucleus)
- [https://github.com/microsoft/hummingbird](https://github.com/microsoft/hummingbird)
- [https://github.com/ptboyer/restful-api-design-tips](https://github.com/ptboyer/restful-api-design-tips)
- [https://github.com/zalando/restful-api-guidelines](https://github.com/zalando/restful-api-guidelines)
- [https://github.com/grafana/k6](https://github.com/grafana/k6)
- [https://github.com/ebhy/budgetml](https://github.com/ebhy/budgetml)
- [https://github.com/bodywork-ml/bodywork-core](https://github.com/bodywork-ml/bodywork-core)
- [https://github.com/ml-tooling/opyrator](https://github.com/ml-tooling/opyrator)
- [https://github.com/triton-inference-server/python_backend](https://github.com/triton-inference-server/python_backend)

### Optimize:

- [https://github.com/ELS-RD/transformer-deploy](https://github.com/ELS-RD/transformer-deploy)
- [https://github.com/yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)
- [https://github.com/Tencent/TurboTransformers](https://github.com/Tencent/TurboTransformers)
- [https://github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [https://github.com/microsoft/fastformers](https://github.com/microsoft/fastformers)
- [https://developer.nvidia.com/blog/boosting-ai-model-inference-performance-on-azure-machine-learning/](https://developer.nvidia.com/blog/boosting-ai-model-inference-performance-on-azure-machine-learning/)

### Explain:

- [https://github.com/marcotcr/anchor](https://github.com/marcotcr/anchor)
- [https://github.com/TeamHG-Memex/eli5](https://github.com/TeamHG-Memex/eli5)
- [https://github.com/QData/TextAttack](https://github.com/QData/TextAttack)
- [https://github.com/PAIR-code/lit](https://github.com/PAIR-code/lit)
- [https://github.com/interpretml/interpret](https://github.com/interpretml/interpret)
- [https://github.com/christophM/interpretable-ml-book](https://github.com/christophM/interpretable-ml-book)
- [https://github.com/cdpierse/transformers-interpret](https://github.com/cdpierse/transformers-interpret)