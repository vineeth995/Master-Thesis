
import streamlit as st
import numpy as np
import os
from catboost import CatBoostRegressor
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import lux
import pickle

# Load model in case model is saved 
#with open('C:/Users/VINEETH/Masterarbeit/Thesis/env/Code/PricePredictionXAI/streamlit-multiapps-master/streamlit-multiapps-master/data/gbt.pkl', 'rb') as file:
    #gbt = pickle.load(file)

# Load model in case model is saved 
#with open('C:/Users/VINEETH/Masterarbeit/Thesis/env/Code/PricePredictionXAI/streamlit-multiapps-master/streamlit-multiapps-master/data/rf.pkl', 'rb') as file:
    #rf = pickle.load(file)    


# Encode variables
def encode(X,categorical_features_indices):
    lbl_enc = LabelEncoder()
    
    print('Inside encode block')

    print(categorical_features_indices)
    if categorical_features_indices is not None:
        for i in categorical_features_indices:
            print(i)
            print(X.iloc[:, i])
            X.iloc[:, i] = lbl_enc.fit_transform(X.iloc[:,i].values)

    return X        
    

# get index of categorical variables from the dataframe
# Source: https://stackoverflow.com/a/38489403/13362641

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
    


# Get X and y variables from inputted dataframe
def variables(x_var,y_var,df):                         
    #X = df[x_var]
    #y = df[y_var.pop()]  
    print('Inside Variables')
    print(x_var,y_var,df.shape)
    if (len(y_var)> 1):
        st.write('Multiple classification not yet supported')
    elif (len(y_var)< 1):
        st.write('Please enter atleast one output variable')
    elif (len(y_var)==1):
        X = df[x_var]
        y = df[y_var]
        print(X.shape,y.shape)
    return X,y,df    
    

# Fit Random Forest model
def randomForestPredict(X,y,n_estimators,max_depth,categorical_features_indices=None):
    print('Reached RandomForestPrecit')
    #lbl_enc = LabelEncoder()
    #print('Initialized Encoder')
    #print(categorical_features_indices)
    #if categorical_features_indices is not None:
        #for i in categorical_features_indices:
            #print(i)
            #print(X.iloc[:, i])
            #X.iloc[:, i] = lbl_enc.fit_transform(X.iloc[:,i].values)
    X=encode(X,categorical_features_indices)
    print(X.head(4))
    rf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X,y)
    y['y_predicted'] = rf.predict(X) 
    final=pd.concat([X, y], axis=1)
    return rf,final


# Fit Gradient Boost model
def gradientBoostPredict(X,y,n_estimators,max_depth,categorical_features_indices=None):
    print('Reached RandomForestPrecit')
    #lbl_enc = LabelEncoder()
    #print('Initialized Encoder')
    #if categorical_features_indices is not None:
        #for i in categorical_features_indices:
            #print(i)
            #print(X.iloc[:, i])
            #X.iloc[:, i] = lbl_enc.fit_transform(X.iloc[:,i].values)
    X=encode(X,categorical_features_indices)
    gb = GradientBoostingRegressor(n_estimators=n_estimators,max_depth=max_depth)
    gb.fit(X,y)
    y['y_predicted'] = gb.predict(X) 
    y['y_predicted'] = gb.predict(X) 
    final=pd.concat([X, y], axis=1)
    return gb,final

# Fit Decision Tree with our criterion
def condensedTree(X,y):
    regressor = DecisionTreeRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    regressor.fit(X_train, y_train)
    print('Inside Condensed Decision Tree')
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    st.write('Decision Tree Model Performance on Training Set:',round(r2_score(y_train,y_pred_train ),2))
    st.write('Decision Tree Model Performance on Test Set:',round(r2_score(y_test,y_pred_test),2))
    #st.write('Decision Tree Model Performance on Training Set (R^2):',0.99)
    #st.write('Decision Tree Model Performance on Test Set (R^2):', 0.70)
    #return regressor,X    

# Fit Surrogate Decision Tree Model
def surrogateModel(X,y):
    regressor = DecisionTreeRegressor(max_depth=20,min_samples_split=3)
    regressor.fit(X, y)
    print('Inside Decision Tree')
    y_pred = regressor.predict(X)
    st.write('Decision Tree Model Performance:',round(r2_score(y,y_pred),2))
    return regressor,X




