# prjctr-ML-in-Prod
[prjctr course Machine Learning in Production](https://prjctr.com/course/machine-learning-in-production)

[Certificate](prjctr-ML-in-Prod/Vadym%20Honcharenko%20Projector%20certificate.pdf)

## Docker 

Got permission denied [trouble fix](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket])

Build docker image:

    docker build --network=host --tag first_docker:latest ./

 --network=host enable to pip install to install packages

Run docker 

    docker run -it --rm --network=host --name test-first_docker first_docker:latest 

--network=host enables to make request to Internet

Before push image to dockerhub don't forget to tag the image  


    docker tag ml_in_prod:week3_latest vadyusikh/ml_in_prod:week3_latest
