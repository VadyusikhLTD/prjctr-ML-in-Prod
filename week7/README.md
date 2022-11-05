# Feast feature store


[Feature store article](https://madewithml.com/courses/mlops/feature-store/)

[Feature store notebook](https://github.com/GokuMohandas/feature-store/blob/main/feature_store.ipynb)

## Instructions on how to setup feast feature store

Create a feature repository at the root of our project. 
Feast will create a configuration file for us and 
we're going to add an additional features.py file 
to define our features.

Run Feast in docker container

    docker build --network=host -t fs:latest .

