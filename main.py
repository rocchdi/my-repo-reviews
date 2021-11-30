from sklearn.feature_extraction.text import CountVectorizer
from check_credentials import check_credentials

from fastapi import FastAPI
from joblib import load
import numpy as np

from fastapi import FastAPI
from fastapi import Request,Query
from fastapi import Header
from fastapi import HTTPException
from typing import Optional,List
from pydantic import BaseModel
import pandas as pd
from enum import Enum
import collections
from random import randint
import json




#api de prediction
apiprediction = FastAPI(
    title='API Prediction',
    description="API de prediction des reviews disney",
    version="1.0.1",
    openapi_tags=[
    {
        'name': 'Home',
        'description': 'Welcome message function'
    },
    {
        'name': 'Prediction',
        'description': 'fonction qui permet de soumettre une phrase  à un modele de prediction au choix parmi plusieurs modeles '
    },
    {
        'name': 'Report',
        'description': 'fonction qui permet d obtenir le rapport des scores pour le modele bow_regression '
    },
    {
        'name': 'Batch',
        'description': 'fonction qui permet de soumettre des reviews en mode batch et obtenir les predictions'
    }

]
    )



#lecture des 4 modèles de prediction: 
# 1 model_bow_regression :  methode Bag of words (BOW)  avec une regression logistique
# 2 model_tfidf_regression :  algo tfidf avec une regression logistique
# 3 model_bow_randomf :  methode Bag of words (BOW) avec Random Forest
# 4 model_tfidf_randomf:  algo tfidf avec RandomForest
#source : https://colab.research.google.com/drive/1GFV6UyaiXWuc3U9cniKeaXAAi4V7y6Kr#scrollTo=If9v0jCDEFKF
#
model1 = load('model_bow_regression.projet')
model2 = load('model_tfidf_regression.projet')
model3 = load('model_bow_randomf.projet')
model4 = load('model_tfidf_randomf.projet')

#lecture du vocabulaire calcule avec CountVectorizer ou TfidfVectorizer
CountVec = load('CountVec.bow')
vect = load('vec.tfidf')


#lecture du fichier classification report du modele "model_bow_regression" dans le dataframe df_scores 
# 
df_report=pd.read_csv('classification_report.csv')
df_report=df_report.iloc[:,1:]


#variables, dicos, classes

class PREDICTIONModel(str, Enum):
    Model1 = "BOW_Regression"
    Model2 = "TFIDF_Regression"
    Model3 = "BOW_RandomForest"
    Model4 = "TFIDF_RandomForest"




#functions
def process_reviews(sentence):

    #vectoriser la review : sentence
    count_vectorizer = CountVectorizer(max_features=2000)
    X_sentence = pd.Series(sentence)
    data = count_vectorizer.transform(X_sentence)

    #utiliser la methode predict du model1
    prediction = model1.predict(data)

    return ' '.join(prediction)








##########################
#routes definition
##########################
@apiprediction.get('/', name='Home', tags=['Home'])
async def get_index():
    """Returns a welcome message
    """
    return {"message": "Je suis prêt à analyser votre review Disney "}




@apiprediction.get("/predict", name='prediction avec un  modele', tags=['prediction'])
def get_predict(request:Request, model:PREDICTIONModel,sentence:str = Query(None), authorization_header: Optional[str] = Header(None, description='Basic username:password')):
    """le  user lance  une prediction avec un modele au choix 
    """


    if check_credentials(request,"all users"):
        #user Authorized

        if model == 'BOW_Regression':
            #le modele attend en input un array de shape n×p
            #utiliser la methode predict du modele
            prediction = model1.predict(CountVec.transform([sentence]))
            print(prediction)
        elif model == 'TFIDF_Regression':
            prediction = model2.predict(vect.transform([sentence]))
        elif model == 'BOW_RandomForest':
            prediction = model3.predict(CountVec.transform([sentence]))
            #data=vectorize_sentence_bow(sentence,CountVec)
            #prediction = model3.predict(data)
        elif model == 'TFIDF_RandomForest':
            prediction = model4.predict(vect.transform([sentence]))
        else :
            raise HTTPException(status_code=404, detail="no selected_model")



        return {
            'prediction': prediction[0],
        }



@apiprediction.get('/report', name='classification report : BOW Regression model', tags=['Classification Report'])
async def get_report():
    """Returns the classification Scores report of the BOW_Regression model 
    """

    res = df_report.to_json(orient='records')[1:-1].replace('},{', '} {')


    return {
        'classification report du BOW_Regression model':res
    }




@apiprediction.get("/batch", name='prediction par  batch : BOW Regression model', tags=['Batch prediction'])
def get_batch(authorization_header: Optional[str] = Header(None, description='Basic username:password')):
    """le  user Admin lance  une prediction avec un fichier batch sur le modele1 : BOW_Regression
    """



    if check_credentials(request,"Administrator"):
        #user Authorized
       

        #lecture fichier batch
        #prediction batch

        res={}

        return {
                'Resultat des predictions batch':res
        }

