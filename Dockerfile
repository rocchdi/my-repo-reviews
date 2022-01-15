FROM ubuntu:20.04

ADD requirements.txt main.py  ./

ADD model_tfidf_regression.joblib model_bow_regression.joblib model_bow_randomf.joblib model_tfidf_randomf.joblib ./

ADD CountVec.joblib vec.joblib df.joblib ./

RUN apt update && apt install python3-pip -y && pip install -r requirements.txt

EXPOSE 8000

CMD uvicorn main:apiprediction --host 0.0.0.0 
