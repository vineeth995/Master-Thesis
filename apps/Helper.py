
import streamlit as st
import numpy as np
import os
from catboost import CatBoostRegressor
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# get index of categorical variables from the dataframe
# Source: https://stackoverflow.com/a/38489403/13362641
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
    
#categorical_features_indices = column_index(X, categorical)



def Variables(x_var,y_var,df):                         
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
    
    #print(X.columns)
    #print(len(X.columns))
    #if ((len(X.columns))>=1):
        
        
    
        
    
       # df=Predict(X,y,categorical_features_indices)
        #return df 
    


def Predict(X,y,categorical_features_indices=None):
    print('Reached Predict')
    #print(categorical_features_indices)
    if categorical_features_indices is not None:
         model = CatBoostRegressor(loss_function = 'MAE',eval_metric = 'RMSE',cat_features=categorical_features_indices)
    else:
         model = CatBoostRegressor(loss_function = 'MAE',eval_metric = 'RMSE') 
            
    
    print('Somehow reached final step')
    print(X.shape,y.shape)
    model.fit( X, y,plot=True )
    
    y['y_predicted'] = model.predict(X) 
    final=pd.concat([X, y], axis=1)
    
    
    print(y.shape)
    return model,final



def RandomForestPredict(X,y,categorical_features_indices=None):
    print('Reached RandomForestPrecit')

    lbl_enc = LabelEncoder()

    print(type(categorical_features_indices))





