# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:50:09 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import pickle

df=pd.read_csv('encoded_disaster.csv')

Data=df.drop(['fema_declaration_string','disaster_number','declaration_date','declaration_title'],axis=1)
Data=Data.drop(['ih_program_declared','ia_program_declared','pa_program_declared','hm_program_declared'],axis=1)
Data=Data.drop(['incident_begin_date','incident_end_date','fips','place_code','designated_area'],axis=1)
Data=Data.drop(['declaration_request_number', 'hash', 'last_refresh', 'id'],axis=1)
print(Data.shape)

#encoding
from sklearn.preprocessing import OrdinalEncoder
columns_to_be_encoded = ['state', 'incident_type']

for column in columns_to_be_encoded:
    # Create Ordinal encoder
    encoder = OrdinalEncoder()
    get_column = Data[column]
    reshaped_vals = get_column.values.reshape(-1, 1)
    # Ordinally encode reshaped_vals
    Data[column] = encoder.fit_transform(reshaped_vals)

# Define the variables
y=Data['declaration_type']
X=Data.drop(['declaration_type'],axis=1)

#Split the Data into Training and Testing sets with test size as 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Modeling
from sklearn.ensemble import RandomForestClassifier
dt_model=RandomForestClassifier(random_state=0)
dt_model.fit(X_train,y_train)
pickle.dump(dt_model,open('disaster_model3.pkl','wb'))
print(dt_model.predict([[12, 1977, 17]]))

