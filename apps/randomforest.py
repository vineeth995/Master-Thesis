import streamlit as st
import pandas as pd
import numpy as np
import os
from data.Helper import variables,column_index,randomForestPredict,surrogateModel,encode,condensedTree,gradientBoostPredict
import pandas as pd
from catboost import CatBoostRegressor
import pandas_profiling
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
#from dataprep.eda import *
#from dataprep.eda import render
#import sweetviz as sv
import webbrowser
#from SessionState import *
import SessionState
#from st_state_patch import *
import st_state_patch
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
import collections
from catboost import CatBoostRegressor,Pool
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import lux





# Main Function
def main():

    st.title("Interpret your Model")

    # upload file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df=load_data(uploaded_file)
 
    # split screen into two columns
    left_column, right_column = st.beta_columns(2)   
    
    # Sample Data
    if left_column.button('View Data'):
        st.write(df)
    
    # Exploratory Data Analysis on Data
    if (right_column.button('Visualize Data') ) :
        eda(df)

    # Display uploaded file
    if uploaded_file:
        display_data(df)
   
# Function to upload data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file,header=0)
    return df

# View Data
# st_shap function referred from https://gist.github.com/andfanilo/6bad569e3405c89b6db1df8acf18df0e
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)      

# Business Logic for creating model and displaying data
def view_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file,header=0)  
        print(df.shape)      
        return df
        


def display_data(df):
    
    # Store input and output variables
    y_var = st.sidebar.multiselect('Select Output Column',df.columns) 
    x_var = st.sidebar.multiselect('Select Columns to include',df.columns)


    # Sanity check for output variables
    if (len(y_var)> 1):
        st.write('Multiple classification not yet supported')
    elif (len(y_var)< 1):
        st.write('Please enter atleast one output variable')
    elif (len(y_var)==1):
        X = df[x_var]
        y = df[y_var]
        print(X.shape,y.shape)

    #initialize Session State storing 
    # Source : https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92    
    session_state = SessionState.get(checkboxed=False,my_value=pd.DataFrame(),button=False,model=RandomForestRegressor(),proc_df=pd.DataFrame(),cat_var=[],y=pd.DataFrame())
    
    # Logic for Next button
    if (st.sidebar.button('Next') or session_state.checkboxed):
        #df_returned = Variables(x_var,y_var,df)       
        session_state.checkboxed= True
        #X,y,df = Variables(x_var,y_var,df)
        cats=st.multiselect('Which of the variables are categorical?:', X.columns)
        print(cats)
        dataframe=pd.DataFrame()
        dataframe=pd.concat([X,y],axis=1)
        #dataframe[cats] =   dataframe[cats].astype(str)
        dataframe.dropna(subset=cats,inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(dataframe[x_var], dataframe[y_var])
        session_state.y = y_test
        #set hyperparameters
        st.write(X_train)
        defaults={'estimators':100,'max_depth':3}
        n_estimators=st.slider('Number of Estimators', min_value=0, max_value=100, value=defaults['estimators'])
        max_depth=st.slider('Max Depth', min_value=0, max_value=100,value=defaults['max_depth'])


        # Logic for Fit Model button
        if (st.button('Fit Model')):
            print('Inside Button')
            
            if cats:
                print('Inside has categorical')
                print(cats)
                print(dataframe.shape)    
                print(dataframe.shape)
                nums = list((collections.Counter(x_var) - collections.Counter(cats)).elements())
                categorical_features_indices = column_index(X_train, cats)
                print(categorical_features_indices)
                print(categorical_features_indices)
                model,df_returned=randomForestPredict(X_train,y_train,n_estimators,max_depth,categorical_features_indices)
                print(df_returned.shape)
                params = model.get_params()
                param_df=pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
                st.write(param_df)
                #for key,value in params.items():
                    #st.write("%s : %s" %(key, value))
                session_state.cat_var=cats
                #explainer = shap.TreeExplainer(model)
                #shap_values = explainer.shap_values(Pool(dataframe[x_var], dataframe[y_var], cat_features=categorical_features_indices))
                #st.pyplot(shap.summary_plot(shap_values, dataframe[x_var], plot_type="bar"))
                st.header('Training Set Results')
                st.write('Root Mean Square Error (RMSE):',round(mean_squared_error(y_train['baseRent'], df_returned['y_predicted'], squared=False),2))
                st.write('Mean Absolute Error (MAE):',round(mean_absolute_error(y_train['baseRent'], df_returned['y_predicted']),2))
                st.write('R-squared score:',round(r2_score(y_train['baseRent'], df_returned['y_predicted']),2))           
                st.write(df_returned)
                X_test=encode(X_test,categorical_features_indices)
                y_test['y_predicted']=model.predict(X_test)
                st.header('Test Set Results')
                st.write('Root Mean Square Error (RMSE):',round(mean_squared_error(y_test['baseRent'],y_test['y_predicted'],squared=False),2))
                st.write('Mean Absolute Error (MAE):',round(mean_absolute_error(y_test['baseRent'],y_test['y_predicted']),2))
                st.write('R-squared score:',round(r2_score(y_test['baseRent'],y_test['y_predicted']),2))  
                testing_final=pd.concat([X_test, y_test], axis=1)
                #df_returned = pd.concat([df_returned,testing_final])
                st.write(testing_final)
                
            else:
                print('Inside has no categorical')
                print(X.shape,y.shape)
                model,df_returned=randomForestPredict(X_train,y_train,n_estimators,max_depth)
                params = model.get_params()
                #for key,value in params.items():
                    #st.write("%s : %s" %(key, value))
                param_df=pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
                st.write(param_df)
                st.header('Training Set Results')
                st.write('Root Mean Squared Error:',round(mean_squared_error(y_train['baseRent'], df_returned['y_predicted'], squared=False),2))
                st.write('Mean Absolute Error:',round(mean_absolute_error(y_train['baseRent'], df_returned['y_predicted']),2))
                st.write('R^2 score:',round(r2_score(y_train['baseRent'], df_returned['y_predicted']),2))           
                st.write(df_returned)
                #X_test=encode(X_test,categorical_features_indices)
                
                
                y_test['y_predicted']=model.predict(X_test)
                st.header('Test Set Results')
                st.write('Root Mean Square Error (RMSE):',round(mean_squared_error(y_test['baseRent'],y_test['y_predicted'],squared=False),2))
                st.write('Mean Absolute Error (MAE):',round(mean_absolute_error(y_test['baseRent'],y_test['y_predicted']),2))
                st.write('R-squared score:',round(r2_score(y_test['baseRent'],y_test['y_predicted']),2))   
                testing_final=pd.concat([X_test, y_test], axis=1)
                st.write(testing_final)
                #df_returned = pd.concat([df_returned,testing_final])

                #explainer = shap.TreeExplainer(model)
                #shap_values = explainer.shap_values(Pool(dataframe[x_var], dataframe[y_var]))
                #st.pyplot(shap.summary_plot(shap_values, dataframe[x_var], plot_type="bar"))
                
            #st.experimental_set_query_params(my_saved_result=df_returned)
            
            session_state.my_value = df_returned
            session_state.proc_df= dataframe
            #print(y_train['baseRent'].shape,df_returned['y_predicted'].shape) 
            session_state.model=model
            
                  
    # Logic for Interpret button  
    if (st.sidebar.button('Interpret') or session_state.button):
        session_state.button= True
        mod=session_state.model
        inp=session_state.proc_df
        inp=inp[1:100]
        cat=session_state.cat_var
        explainer = shap.TreeExplainer(mod)
        df_returned=session_state.my_value 
        print('This is what we get')   
        df_returned = df_returned.reset_index(drop=True)
        st.write(df_returned) 
        #st.write(mod)
        explain_row = st.selectbox('Choose row you want to interpret',df_returned.index)
 

        if len(df_returned) != 0:
            #st.write(df_returned)
            if (st.button('Show Interpretability Plots')):
                if len(cat)!=0:
                    st.write('You have chosen to interpret row:',explain_row) 
                    shap_values = explainer.shap_values(df_returned[x_var][1:100])
                    st.set_option('deprecation.showPyplotGlobalUse', False)

                    
                    

                    # Random Forest Feature Importances
                    st.title('Feature Importance Plot (MDI)')
                    features = inp.columns
                    importances = mod.feature_importances_
                    indices = np.argsort(importances)

                    plt.title('Feature Importances')
                    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Relative Importance')
                    st.pyplot(plt.show())

                    # FI calculation
                    FI=[features[i] for i in indices]
                    FI.reverse()
                    print(FI)
                    FI = FI[0:7]
                    st.write(FI)

                    
                    ###

                    # Permutation Feature Importances
                   
                    st.title('Permutation based Feature Importance')   
                                  
                    perm = PermutationImportance(mod, random_state=1).fit(df_returned[x_var], df_returned[y_var])
                    #eli5.show_weights(perm, feature_names = X_train.columns.tolist())
                    html_object =eli5.show_weights(perm, feature_names = df_returned[x_var].columns.tolist())

                    w = eli5.show_weights(perm, feature_names=X_train.columns.tolist())
                    result = pd.read_html(w.data)[0]
                    result['wt']=result['Weight'].apply(lambda x:float(x.split(' ')[0]))
                    plt.title('Permutation based Feature Importance')
                    plt.barh(result['Feature'], result['wt'], color='g', align='center')
                    #plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Permutation Importance')
                    plt.gca().invert_yaxis()
                    st.pyplot(plt.show())

                    
                    raw_html = html_object._repr_html_()
                    #components.html(raw_html)
                    
                    # PFI calculation
                    w = eli5.show_weights(perm, feature_names=X_train.columns.tolist())
                    result = pd.read_html(w.data)[0]
                    top_7=result['Feature'].head(7)
                    PFI=top_7.to_list()
                    st.write(PFI)

                    ####

                    # SHAP Interpretations
                    st.title('SHAP Interpretations')
                    st.write('SHAP-based Feature Importance Plot')
                    st.pyplot(shap.summary_plot(shap_values, df_returned[x_var], plot_type="bar"))
                    # SV calculation
                    shap_sum = np.abs(shap_values).mean(axis=0)
                    importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
                    importance_df.columns = ['column_name', 'shap_importance']
                    importance_df = importance_df.sort_values('shap_importance', ascending=False)
                    importance_df['column_name'].head(7)
                    list_of_sv = importance_df['column_name'].head(7)
                    SV = list_of_sv.to_list()
                    st.write(SV)
                    st.write('SHAP Local Force Plot for the chosen observation')
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[explain_row,:], df_returned[x_var].iloc[explain_row,:]))
                    #print(shap_values.shape,df_returned[x_var].shape)
                    st.write('SHAP Summary Plot')
                    st.pyplot(shap.summary_plot(shap_values, df_returned[x_var][1:100]))
                    st.write('SHAP Global Force Plot')
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[1:100], df_returned[x_var][1:100]),height=400)

                    

                    Final = list(set(FI) & set(PFI) & set(SV))

                    st.title('The important variables are:')
                    st.write(Final)


                    # Condensed Decision Tree
                    
                    st.write('Decision Tree on Important Variables')
                    condensedTree(df_returned[x_var],df_returned[y_var])
                    st.write('Decision Tree on Important Variables')
                    condensedTree(df_returned[x_var],df_returned[y_var])


                    # Surrogate Model Analysis
                    st.title('Surrogate Model Analysis')
                    tree_model,X_tree=surrogateModel(df_returned[x_var],df_returned['y_predicted'])
                    explainer_dt = shap.TreeExplainer(tree_model)
                    shap_values_dt = explainer.shap_values(X_tree)
                    st.write('Shapley Values for the Surrogate Decision Tree')
                    st.pyplot(shap.summary_plot(shap_values_dt, X_tree, plot_type="bar"))


                    

 
                else:
                    
                    # Random Forest Feature Importances
                    st.title('Random Forest Feature Importance Plot')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    features = X.columns
                    importances = mod.feature_importances_
                    indices = np.argsort(importances)

                    plt.title('Feature Importances')
                    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Relative Importance')
                    st.pyplot(plt.show())


                    # FI calculation
                    FI=[features[i] for i in indices]
                    FI.reverse()
                    FI = FI[0:6]
                    st.write(FI)


                    ###
                    # Permutation Importances 
                    st.title('Random Forest Permutation Importance Plot')                   
                    perm = PermutationImportance(mod, random_state=1).fit(inp[x_var], inp[y_var])
                    #eli5.show_weights(perm, feature_names = X_train.columns.tolist())
                    html_object =eli5.show_weights(perm, feature_names = inp[x_var].columns.tolist()) 
                    raw_html = html_object._repr_html_()
                    #components.html(raw_html)

                    w = eli5.show_weights(perm, feature_names=X_train.columns.tolist())
                    result = pd.read_html(w.data)[0]
                    result['wt']=result['Weight'].apply(lambda x:float(x.split(' ')[0]))
                    plt.title('Permutation based Feature Importance')
                    plt.barh(result['Feature'], result['wt'], color='g', align='center')
                    #plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Permutation Importance')
                    plt.gca().invert_yaxis()
                    st.pyplot(plt.show())


                    # PFI calculation
                    w = eli5.show_weights(perm, feature_names=X_train.columns.tolist())
                    result = pd.read_html(w.data)[0]
                    top_7=result['Feature'].head(7)
                    PFI=top_7.to_list()
                    st.write(PFI)


                    
                    ####
                    st.write('You have chosen to interpret row:',explain_row) 
                    shap_values = explainer.shap_values(X[1:100])
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    # Permutation Feature Importances


                    # SHAP Interpretations
                    st.title('Interpretations using SHAP')
                    st.write('SHAP-based Feature Importance Plot')
                    st.pyplot(shap.summary_plot(shap_values, df_returned[x_var], plot_type="bar"))
                    # SV calculation
                    shap_sum = np.abs(shap_values).mean(axis=0)
                    importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
                    importance_df.columns = ['column_name', 'shap_importance']
                    importance_df = importance_df.sort_values('shap_importance', ascending=False)
                    importance_df['column_name'].head(7)
                    list_of_sv = importance_df['column_name'].head(7)
                    SV = list_of_sv.to_list()
                    st.write(SV)
                    st.write('SHAP Local Force Plot for the chosen observation')
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[explain_row,:], df_returned.iloc[explain_row,:]))
                    st.write('SHAP Summary Plot')
                    st.pyplot(shap.summary_plot(shap_values, df_returned[x_var][1:100]))
                    st.write('SHAP Global Force Plot')
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[1:100], df_returned[x_var][1:100]),height=400)


                    


                    Final = list(set(FI) & set(PFI) & set(SV))
                    
                    st.title('The important variables are:')
                    st.write(Final)

                    # Condensed Decision Tree
                    st.write('Decision Tree on Important Variables')
                    condensedTree(df_returned[Final],df_returned[y_var])
                    st.write('Decision Tree on All Variables')
                    condensedTree(df_returned[x_var],df_returned[y_var])

                    
                    # Surrogate Model Analysis
                    st.title('Surrogate Model Analysis')
                    tree_model,X_tree=surrogateModel(df_returned[x_var],df_returned['y_predicted'])
                    explainer_dt = shap.TreeExplainer(tree_model)
                    shap_values_dt = explainer.shap_values(X_tree)
                    st.write('Shapley Values for the Surrogate Decision Tree')
                    st.pyplot(shap.summary_plot(shap_values_dt, X_tree, plot_type="bar"))


                    

                    

def eda(df):
    #profile = ProfileReport(df, title="Analysis of your Dataset",minimal=True)
    profile = ProfileReport(df, title="Analysis of your Dataset")
    components.html(profile.to_html(),height=10000)


if __name__ == "__main__":
    main()   
    
    
    
    



      


