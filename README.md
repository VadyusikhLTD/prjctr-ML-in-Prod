# prjctr-ML-in-Prod
[prjctr course Machine Learning in Production](https://prjctr.com/course/machine-learning-in-production)

Build docker image:

    docker build --network=host --tag first_docker:latest ./

 --network=host enable to pip install to install packages

Run docker 

    docker run -it --rm --network=host --name test-first_docker first_docker:latest 

--network=host enables to make request to Internet
