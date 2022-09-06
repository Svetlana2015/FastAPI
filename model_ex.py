b# 1. Library imports
#import pickle
import joblib

import pandas as pd 
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier



# Load data and save indices of columns
df = pd.read_csv("train.csv")
#features = df.drop('TARGET', axis=1)
#joblib.dump(features, 'features.joblib')

# Fit and save an OneHotEncoder
categorical = df.select_dtypes(include = ['object'])
enc = OneHotEncoder().fit(categorical)
joblib.dump(enc, 'encoder_2.joblib')

# Transform variables, merge with existing df and keep column names

column_names = enc.get_feature_names_out(['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE',
                  'FLAG_OWN_REALTY', 'FLAG_OWN_CAR','NAME_EDUCATION_TYPE',  'OCCUPATION_TYPE',
                 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START'])
array = enc.transform(categorical).toarray()
encoded_variables = pd.DataFrame(array,index = categorical.index, columns=column_names)
df = df.drop(['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE',
                  'FLAG_OWN_REALTY', 'FLAG_OWN_CAR','NAME_EDUCATION_TYPE',  'OCCUPATION_TYPE',
                 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START'], axis = 1)
df = pd.concat([df, encoded_variables], axis=1)
print(df)


# Fit and save an MinMaxScaler
numeric=df.select_dtypes(exclude=['object'])
numeric_2=numeric.drop('TARGET', axis=1)

scaler = MinMaxScaler().fit(numeric_2)
joblib.dump(scaler,'mmsc_3.joblib')

# Transform variables, merge with existing df and keep column names

numeric_2[numeric_2.columns] = pd.DataFrame(scaler.transform(numeric_2), index= numeric_2.index)
numeric=pd.concat([numeric_2, numeric.TARGET], axis=1)
df=numeric


# Fit and save model
# RandomForestClassifier
X = df.drop('TARGET', axis=1)
y = df['TARGET']
random_forest=RandomForestClassifier(max_depth=127,
                          min_samples_split=32,
                          n_estimators=250,
                          max_features= 'sqrt')
clf = random_forest.fit(X, y)
joblib.dump(clf, 'mrf.joblib')






