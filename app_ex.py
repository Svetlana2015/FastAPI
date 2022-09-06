# Data Handling
import joblib

import numpy as np
import pandas as pd


# Server
import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse


app = FastAPI()

# Initialize files
clf = joblib.load('mrf.joblib')
enc = joblib.load('encoder_2.joblib')
scaler = joblib.load('mmsc_3.joblib')
#features = joblib.load('features.joblib')

# Class which describes a single utilisateur
class Parametres(BaseModel):
    
    DAYS_BIRTH: int = Field(title = "Âge du client en jours au moment de la demande", ge=-30000, le=-6570)
    
    CNT_CHILDREN: int = Field (title = "Nombre d'enfants", ge = 0)
    CNT_FAM_MEMBERS: int = Field (title = "Combien de membres de la famille le client a-t-il", ge=0)
    
    
    FLAG_PHONE: int = Field(title = "Le client a-t-il fourni un téléphone résidentiel (1=OUI, 0=NON)", list = [1,0])
    FLAG_CONT_MOBILE : int = Field(title = "Le téléphone portable était-il joignable (1=OUI, 0=NON)", list = [1,0])
    DAYS_LAST_PHONE_CHANGE: int = Field(title = "Combien de jours avant l'application le client a-t-il changé de téléphone", ge=0, le=1000)
    FLAG_EMAIL : int = Field(title = "Le client a-t-il fourni un e-mail (1=OUI, 0=NON)", list = [1,0])
    
    DAYS_EMPLOYED: int = Field(title = "Combien de jours avant la demande la personne a commencé l'emploi actuel?", ge=-18000, le=0)
    FLAG_WORK_PHONE : int = Field(title = "Le client a-t-il fourni un téléphone portable fonctionnel", list=[1,0])
    FLAG_EMP_PHONE : int = Field(title = "Le client a-t-il fourni un téléphone fixe fonctionnel (1=OUI, 0=NON)", list=[1,0])
    LIVE_CITY_NOT_WORK_CITY : int = Field(title = "Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle 1=différent, 0=identique, au niveau de la ville)",
                                          list = [1,0])
    LIVE_REGION_NOT_WORK_REGION : int = Field(title = "Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la ville)",
                                                       list = [1,0])
    REG_CITY_NOT_WORK_CITY : int = Field(title = "Signaler si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la ville)",
                                                  list = [1,0])
    REG_CITY_NOT_LIVE_CITY : int = Field(title = "Signaler si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent, 0=identique, au niveau de la ville)",
                                                  list = [1,0])
    REG_REGION_NOT_WORK_REGION : int = Field(title = "Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la région)",
                                                      list = [1,0])
    DAYS_ID_PUBLISH: int = Field(title = "Combien de jours avant le dépôt de la demande le client a-t-il changé la pièce d'identité avec laquelle il a demandé le service?",
                                        ge = -8000, le = 0)
    DAYS_REGISTRATION: int = Field(title = "Combien de jours avant la demande le client a-t-il modifié son inscription?", ge = -25000, le = 0)


    AMT_GOODS_PRICE: float = Field(title = "Pour les prêts à la consommation c'est le prix des biens pour lesquels le prêt est accordé", ge = 10000, le = 5000000) 
    AMT_ANNUITY: float = Field(title = "Rente de prêt", ge = 1000, le = 300000)
    AMT_CREDIT: float = Field(title = "Montant du crédit du prêt", ge = 30000,le = 4500000)
    AMT_INCOME_TOTAL : float = Field(title = "Revenu du client", ge = 10000, le = 17000000)
    AMT_REQ_CREDIT_BUREAU_YEAR: int = Field(title = "Nombre de demandes de renseignements au bureau de crédit concernant le client", ge = 0, le = 30)
    HOUR_APPR_PROCESS_START: int = Field(title = "À quelle heure environ le client a-t-il demandé le prêt", ge = 0, le = 24)
    
    REGION_RATING_CLIENT_W_CITY : int = Field(title = "Notre évaluation de la région où vit le client en tenant compte de la ville (1,2,3)", list = [1,2,3], ge = 1, le = 3)
    REGION_RATING_CLIENT : int = Field(title = "Notre évaluation de la région où vit le client (1,2,3)", list = [1,2,3], ge = 1, le = 3)
    REGION_POPULATION_RELATIVE : float = Field (title = "Population normalisée de la région dans laquelle vit le client (un nombre plus élevé signifie que le client vit dans une région plus populaire)",
                                                ge=0, le = 0.1)
    FLAG_DOCUMENT_21 : int = Field(title = "Le client a-t-il fourni le document 21 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_20 : int = Field(title = "Le client a-t-il fourni le document 20 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_19 : int = Field(title = "Le client a-t-il fourni le document 19 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_18 : int = Field(title = "Le client a-t-il fourni le document 18 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_17 : int = Field(title = "Le client a-t-il fourni le document 17 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_16 : int = Field(title = "Le client a-t-il fourni le document 16 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_15 : int = Field(title = "Le client a-t-il fourni le document 15 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_14 : int = Field(title = "Le client a-t-il fourni le document 14 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_13 : int = Field(title = "Le client a-t-il fourni le document 13 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_11 : int = Field(title = "Le client a-t-il fourni le document 11 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_9 : int = Field(title = "Le client a-t-il fourni le document 9 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_8 : int = Field(title = "Le client a-t-il fourni le document 8 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_7 : int = Field(title = "Le client a-t-il fourni le document 7 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_6 : int = Field(title = "Le client a-t-il fourni le document 6 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_5 : int = Field(title = "Le client a-t-il fourni le document 5 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_3 : int = Field(title = "Le client a-t-il fourni le document 3 (1=OUI, 0=NON)", list = [1,0])
    FLAG_DOCUMENT_2 : int = Field(title = "Le client a-t-il fourni le document 2 (1=OUI, 0=NON)", list = [1,0])
    OBS_60_CNT_SOCIAL_CIRCLE: int = Field(title = "Combien d'observations de l'environnement social du client avec un défaut observable de 30 DPD (jours de retard)",
                                          list = [0,1,2,3,4,5],ge = 1,le = 5)
    OBS_30_CNT_SOCIAL_CIRCLE: int = Field(title = "Combien d'observations de l'environnement social du client avec un défaut observable de 60 DPD (jours de retard)",
                                          list = [0,1,2,3,4,5],ge = 1,le = 5)
    EXT_SOURCE_2: float = Field(title = "Score normalisé à partir d'une source de données externe (2)", ge = 0.0, le = 1.0)
    EXT_SOURCE_3: float = Field(title = "Score normalisé à partir d'une source de données externe (3)", ge = 0.0, le = 1.0) #description="The score must be greater than zero")

    CODE_GENDER: str = Field (title = "Sexe", list = ["F","M"]) #List[str] = Query(default=['F', 'M'])
    NAME_FAMILY_STATUS:str = Field (title = "Situation familiale du client", list = ['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated'])
    NAME_TYPE_SUITE : str = Field(title = "Qui accompagnait le client lors de sa demande de prêt?", list = ['Unaccompanied', 'Family', 'Spouse, partner', 'Children',
                                           'Other_A', 'Other_B', 'Group of people'])
    NAME_HOUSING_TYPE : str = Field(title = "Quelle est la situation de logement du client", list = ['House / apartment', 'Rented apartment', 'With parents',
                                                                                               'Municipal apartment', 'Office apartment', 'Co-op apartment'])
    FLAG_OWN_REALTY : str = Field(title = "Signaler si le client possède une maison ou un appartement (Y=OUI, N=NON)", list = ['Y','N'])
    FLAG_OWN_CAR : str = Field(title = "Signaler si le client possède une voiture (Y=OUI, N=NON)", list = ['Y','N'])
    NAME_EDUCATION_TYPE : str = Field(title = "Niveau de scolarité le plus élevé atteint par le client", list = ['Secondary / secondary special', 'Higher education',
                                                                                                           'Incomplete higher', 'Lower secondary', 'Academic degree'])
    OCCUPATION_TYPE : str = Field(title = "Quel type d'occupation le client a-t-il", list = ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers',
                                                                                             'Sales staff',
                                                                                       'Cleaning staff', 'Cooking staff','Private service staff', 'Medicine staff',
                                                                                       'Security staff', 'High skill tech staff', 'Waiters/barmen staff',
                                                                                       'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'])

    NAME_INCOME_TYPE : str = Field(title = "Type de revenu des clients", list = ['Working', 'State servant', 'Commercial associate', 'Pensioner',
                                                                                 'Student', 'Businessman', 'Maternity leave'])
    NAME_CONTRACT_TYPE : str = Field(title = "Identification si le prêt est en espèces ou renouvelable", list = ['Cash loans', 'Revolving loans'])
    WEEKDAY_APPR_PROCESS_START : str = Field(title = "Quel jour de la semaine le client a-t-il demandé le prêt", list = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
                                                                                                           'SATURDAY', 'SUNDAY'])
    

               
@app.post("/predict")
def predict(datas: Parametres):
    
    # Extract data in correct order
    datas_dict = datas.dict()
    print(datas_dict)

    df = pd.DataFrame([datas_dict])
    print(df)
    print(df.info())
   

    # Apply one-hot encoding
    categorical = df.select_dtypes(include = ['object'])
    column_names = enc.get_feature_names_out(['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE',
                  'FLAG_OWN_REALTY', 'FLAG_OWN_CAR','NAME_EDUCATION_TYPE',  'OCCUPATION_TYPE',
                 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START'])
    array = enc.transform(categorical).toarray()
    encoded_features = pd.DataFrame(array,index = categorical.index, columns=column_names)
    df = df.drop(['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE',
                  'FLAG_OWN_REALTY', 'FLAG_OWN_CAR','NAME_EDUCATION_TYPE',  'OCCUPATION_TYPE',
                 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START'], axis = 1)
    df = pd.concat([df, encoded_features], axis=1)
    print("done")

    # Apply MinMaScaler
    df[df.columns] = pd.DataFrame(scaler.transform(df), index= df.index)
    print('done')

    # Create and return prediction

    prediction = clf.predict(df)
    prediction = prediction.tolist()
    #probability = clf.predict_proba(df)
    probability = clf.predict_proba(df).max()

    return {'prediction': prediction[0],
           'probability': probability,
            }



    
