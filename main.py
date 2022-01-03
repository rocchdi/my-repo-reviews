from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from check_credentials import check_credentials


from fastapi import FastAPI
from joblib import load
import numpy as np

from fastapi import FastAPI , Depends
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

from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from fastapi import FastAPI, File, UploadFile
import shutil



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
        'name': 'prediction',
        'description': 'fonction qui permet de soumettre une phrase  à un modele de prediction au choix parmi plusieurs modeles '
    },
    {
        'name': 'report',
        'description': 'fonction qui permet d obtenir le rapport des scores pour le modele bow_regression '
    },
    {
        'name': 'identification',
        'description': 'Identification de l utilisateur , pour pouvoir utiliser l API'
    },
    {
        'name': 'report/relance',
        'description': 'Permet de relancer l entrainement des données , pour vérifier le resultat des scores des modeles  '
    },
    {
        'name': 'enregistrement',
        'description': 'Permet d importer un fichier csv , avec 1 seul colonne qui contient les phrases à analyser   '
    },

]
    )

# ========== Sécurité ===========

# création du schema de sécurité
security = HTTPBasic()

#Initialisation du compte sécurisé
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "administrateur")
    correct_password = secrets.compare_digest(credentials.password, "password")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


#lecture des 4 modèles de prediction: 
# 1 model_bow_regression :  methode Bag of words (BOW)  avec une regression logistique
# 2 model_tfidf_regression :  algo tfidf avec une regression logistique
# 3 model_bow_randomf :  methode Bag of words (BOW) avec Random Forest
# 4 model_tfidf_randomf:  algo tfidf avec RandomForest
#source : https://colab.research.google.com/drive/1GFV6UyaiXWuc3U9cniKeaXAAi4V7y6Kr#scrollTo=If9v0jCDEFKF

model1 = load('model_bow_regression.joblib')
model2 = load('model_tfidf_regression.joblib')
model3 = load('model_bow_randomf.joblib')
model4 = load('model_tfidf_randomf.joblib')


#lecture du vocabulaire calcule avec CountVectorizer ou TfidfVectorizer
CountVec = load('CountVec.joblib')
vect = load('vec.joblib')


#lecture du fichier classification report du modele "model_bow_regression" dans le dataframe df_scores 
# 
df_report=pd.read_csv('classification_report.csv')
df_report=df_report.iloc[:,1:]



#variables, dicos, classes

class PredictionModel(str, Enum):
    Model1 = "BOW_Regression"
    Model2 = "TFIDF_Regression"
    Model3 = "BOW_RandomForest"
    Model4 = "TFIDF_RandomForest"




#functions
def process_reviews(sentence):

    #vectoriser la review : sentence
    count_vectorizer = CountVectorizer(max_features=2000)
    x_sentence = pd.Series(sentence)
    data = count_vectorizer.transform(x_sentence)

    #utiliser la methode predict du model1
    prediction = model1.predict(data)
    print(prediction)
    return ' '.join(prediction)





##########################
#routes definition
##########################
@apiprediction.get('/', name='Home', tags=['Home'])
async def get_index(username: str = Depends(get_current_username)):
    """Returns a welcome message
    """
    return {"message": "Je suis prêt à analyser votre review Disney "}


# ========== Page pour s'indentifier ( valable pendant toute la navigation ) pour changer => fermer le navigateur ==========
@apiprediction.get("/identification" , name='identification utilisateur', tags=['identification'])
def read_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    return {"username": credentials.username, "password": credentials.password}


# ========== Prédition pour une phrase ==========
@apiprediction.get("/predict", name='prediction avec un  modele', tags=['prediction'])
def get_predict(request:Request, model:PredictionModel,sentence:str = Query(None),username: str = Depends(get_current_username)):
    """le  user lance  une prediction avec un modele au choix 
    """

    if model == 'BOW_Regression':
        #le modele attend en input un array de shape n×p
        #utiliser la methode predict du modele
        prediction = model1.predict(CountVec.transform([sentence]))
        proba = model1.predict_proba(CountVec.transform([sentence]))


    elif model == 'TFIDF_Regression':
        prediction = model2.predict(CountVec.transform([sentence]))
        proba = model2.predict_proba(CountVec.transform([sentence]))
    elif model == 'BOW_RandomForest':
        prediction = model3.predict(CountVec.transform([sentence]))
        proba = model3.predict_proba(CountVec.transform([sentence]))
        #data=vectorize_sentence_bow(sentence,CountVec)
        #prediction = model3.predict(data)
    elif model == 'TFIDF_RandomForest':
        prediction = model4.predict(CountVec.transform([sentence]))
        proba = model4.predict_proba(CountVec.transform([sentence]))
    else :
        raise HTTPException(status_code=404, detail="pas de modéle séléctionné")
    
    return {'prediction' :prediction[0] , 'Probabilité négatif': proba[0][0] ,'Probabilité neutre': proba[0][1] ,'Probabilité positif': proba[0][2] }

# ========== Pour voir les scores des différents modéles ==========
@apiprediction.get('/report', name='classification report : BOW Regression model', tags=['report'])
async def get_report(model:PredictionModel ,username: str = Depends(get_current_username)):

    try:
        # Ouvre le fichier score.json en lecture
        with open("score.json") as fichier:
            jsonData = json.load(fichier)

        #print(jsonData)
        fichier.close

        if model == 'BOW_Regression':
            #Affiche le valeur 'Bow_Regression' du fichier .json
            res = jsonData['model']['Bow_Regression']

        elif model == 'TFIDF_Regression':
            res = jsonData['model']['TFIDF_Regression']

        elif model == 'BOW_RandomForest':
            res = jsonData['model']['BOW_RandomForest']

        elif model == 'TFIDF_RandomForest':
            res = jsonData['model']['TFIDF_RandomForest']

        else :
            raise HTTPException(status_code=404, detail="pas de modéle séléctionné")

        return {"Score_model" : res}

    except Exception as e:
        #Retourne une erreur si le bloc "try" ne ce fini pas correctement
        print("Erreur : Lecture du fichier score.json  " , e )
 
# ========== Pour relancer l'entrainement des modéles et vérifier le score ==========
@apiprediction.get('/report/relance', name='Relance l entrainement des modeles', tags=['report/relance'])
async def get_report(model:PredictionModel ,username: str = Depends(get_current_username)):
       
    # ----- Initialisation des variable -----
    #Df deja traité est enregistré avec joblib
    df  = load('df.joblib')

    #----- Séparation des données en jeu d'entraînement et de test -----

    features = df['Review_Text']
    target = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(features, target)
    vectorizer = CountVec
    X_train_cv = vectorizer.fit_transform(X_train)
    X_test_cv = vectorizer.transform(X_test)

    # ----- Entrainement des models  -----

    if model == 'BOW_Regression':
        model1.fit(X_train_cv, y_train)
        res = model1.score(X_test_cv, y_test)

    elif model == 'TFIDF_Regression':
        res = jsonData['model']['TFIDF_Regression']
        model2.fit(X_train_cv, y_train)
        res = model2.score(X_test_cv, y_test)
    
    elif model == 'BOW_RandomForest':
        res = jsonData['model']['BOW_RandomForest']
        model3.fit(X_train_cv, y_train)
        res = model3.score(X_test_cv, y_test)

    elif model == 'TFIDF_RandomForest':
        res = jsonData['model']['TFIDF_RandomForest']
        model4.fit(X_train_cv, y_train)
        res = model4.score(X_test_cv, y_test)

    else :
        raise HTTPException(status_code=404, detail="pas de modéle séléctionné")

    return {"Score_model" : res}

# ========== Enregistrement d'un fichier csv avec des phrases ==========

@apiprediction.post("/enregistrement/",name='Enregistrement fichier.csv', tags=['enregistrement'])
async def create_upload_file(model:PredictionModel  , file: UploadFile = File(...),username: str = Depends(get_current_username)):
    
    # -----Enregistre le fichier -----
    #Ouvre un fichier fichier_import.csv à la racine du projet => copie le fichier.csv importé dedans 
    with open("fichier_import.csv","wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    buffer.close()    
    
    # ----- Lit le fichier csv  -----
    df = pd.read_csv("fichier_import.csv",sep=';',names = ['col1'])
    
    resultat= []
    # ----- Pour chaque ligne on calcule le score de la phrase -----
    i=0
    for sentence in df['col1']:
        
        if model == 'BOW_Regression':
            prediction = model1.predict(CountVec.transform([sentence]))
     
        elif model == 'TFIDF_Regression':
            prediction = model2.predict(CountVec.transform([sentence]))

        elif model == 'BOW_RandomForest':
            prediction = model3.predict(CountVec.transform([sentence]))

        elif model == 'TFIDF_RandomForest':
            prediction = model4.predict(CountVec.transform([sentence]))

        else :
            raise HTTPException(status_code=404, detail="pas de modéle séléctionné")

        resultat.append("sentence :"+str(i)+str(prediction))
        i =i+1
        
    return resultat

