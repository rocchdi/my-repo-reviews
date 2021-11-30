FROM ubuntu:20.04

ADD requirements.txt main.py  check_credentials.py ./

ADD model_tfidf_regression.projet model_bow_regression.projet model_bow_randomf.projet model_tfidf_randomf.projet ./

ADD CountVec.bow vec.tfidf ./

ADD classification_report.csv ./ 

RUN apt update && apt install python3-pip -y && pip install -r requirements.txt

EXPOSE 8000

CMD uvicorn main:apiprediction --host 0.0.0.0 
