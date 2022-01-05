FROM debian:latest

RUN apt-get update && apt-get install python3-pip -y && pip3 install requests
ADD predict_test1.py /home/work/predict_test1.py
WORKDIR /home/work
VOLUME /home/logs
CMD python3 predict_test1.py 

