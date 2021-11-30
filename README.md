# my-repo-reviews
My repository for reviews project 2


# Overview
A Fast API to predict sentiment Analysis using differents algo/models , with kubernetes deployment and a testing environment with DockerCompose

# The Fast API image
The FAst API image is already created and puched to the DockerHub repository : rocchdi/apipredict:1.0.1
You can check the Dockerfile of the API and the requirements.txt used to create the image
You can also check the API code


# How to use the Fast API with kubernetes
##Deployment and running with kubernetes
use the kubernetes yml files to deploy the API and run it using the following in your kubernetes environment:

```
kubectl create -f  my-deployment-project.yml
kubectl create -f  my-service-project.yml
kubectl create -f  my-ingress-project.yml
```

use the following port redirection before connection to the API
ssh -i data_enginering_machine.pem  -L 80:192.168.49.2:80 ubuntu@<YOUR_IP_MACHINE>

connect the API with your navigator
http://localhost:80/docs

choose the endpoint : /predict
to test the API, you can use the following parameters :
model: BOW_Regression
sentence: hello disney
authorization-header: Basic alice:wonderland

you can also check the other endpoints :
/report : to view the classification report from the BOW model
/batch  : to predict sentiment from  a csv file (not implemented yet) 



#The testing environment of the API using docker-compose

you can find in the setup.sh files the cammands used to create the docker test image and how to run docker-compose
the test is called : predict_test1, it  checks a complete request to the /predict endpoint

