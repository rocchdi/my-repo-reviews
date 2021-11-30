# my-repo-reviews
My repository for reviews project 2


# Overview
A Fast API to predict sentiment Analysis using differents algo/models , with kubernetes deployment and a testing environment with DockerCompose
the API used a Basic authentication (will be using a BAsic base64 encoded password in next release)

# The Fast API image
The FAst API image is already created and puched to the DockerHub repository : rocchdi/apipredict:1.0.1
You can check the Dockerfile of the API and the requirements.txt used to create the image
You can also check the API code (main.py)


# How to use and test the Fast API in your machine using a python virtual environment

-create a new python virtual environment
-install the requirements :  pip install -r requirements.txt
-copy : all the .py files (the code), the .project files (prediction models), the vect.tfidf and countvec.bow files (vocabularies files)
and also the .csv file (classification report) in your machine.
-to run the API : uvicorn main:apiprediction --relaod
-to test the API: redirect the port 8000, and use the url : http://localhost:8000/docs

choose the endpoint : /predict
to test the API, you can use the following parameters :
model: BOW_Regression
sentence: hello disney
authorization-header: Basic alice:wonderland

you can also check the other endpoints :
/report : to view the classification report from the BOW model
/batch  : to predict sentiment from  a csv file (not implemented yet)



# How to use the Fast API with kubernetes
##Deployment and running with kubernetes
use the kubernetes yml files to deploy the API and run it using the following in your kubernetes environment:

```
kubectl create -f  my-deployment-project.yml
kubectl create -f  my-service-project.yml
kubectl create -f  my-ingress-project.yml
```

use the following port redirection :
ssh -i data_enginering_machine.pem  -L 80:192.168.49.2:80 ubuntu@<YOUR_IP_MACHINE>

connect the API with your navigator : 
http://localhost:80/docs



# How to use The testing environment of the API with docker-compose

In the test folder you can find a setup.sh file that contains the cammands that allow you  create the docker test image and run docker-compose.
The test is a predict test, it tests  a complete request to the /predict endpoint and display SUCESS if it receives the http code 200.
In the docker-compose the test container should run after the API is up. We use a restart on failure to run the test when the api is not ready.
Other tests will be implemented soon.

