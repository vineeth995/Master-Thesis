import streamlit as st
from multiapp import MultiApp
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from apps import  gradientboost,randomforest 

# Initializing Multiapp instance
# Source: https://github.com/upraneelnihar/streamlit-multiapps/blob/master/multiapp.py 
apps = MultiApp()

# Routing Pages to respective models
apps.add_app("Regression with GradientBoostingRegressor", gradientboost.main)
apps.add_app("Regression with RandomForestRegressor", randomforest.main)

# Initialize Run
apps.run()