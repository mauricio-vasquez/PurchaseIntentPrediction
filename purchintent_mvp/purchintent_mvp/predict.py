# # APZIVA AI RESIDENCY PROGRAM
# 
# ## Project 2: Intent Marketing - MINIMUM VIABLE PRODUCT ##
# ### Lead prediction system for the banking business

# ### Prepared by: Mauricio Vásquez A. 
# ### Mentor: Swarnabha Ghosh. 
# ### Last updated on: September 2023
# 
# #### Contact email: mauricio_vasquez_andrade@hotmail.com
# #### LinkedIn: https://www.linkedin.com/in/mauricio-vasquez-andrade-ecuador/


#### MAIN TRAINING SCRIPT ####

# Install libraries
#pip install numpy sklearn imblearn shap catboost pathlib

#Import libraries
import pandas as pd
import numpy as np
import math
import usersetts as setts
from pickle import load
import streamlit as st


# Import preprocessing
import etl

def cleandata(df):  
    # Data processing 
    ### Ordinal features
    df = etl.ordinalcopier(df)   
    df = etl.ord_imputer(df, load_imputer = True)
    df = etl.ord_encode(df, load_encod = True)

    ### Nominal features
    df = etl.nom_imputer(df, load_imputer = True)
    df = etl.one_hot_encode(df)

    ### Numeric features
    df = etl.scale_numeric_columns(df)
    return df

def leadprediction(df):
    model = load(open('catboostclassifier.pkl','rb'))
    # Predict values
    y_pred = model.predict(df)
    # Predict probabilities
    y_prob_pred = model.predict_proba(df)[:,1]
    # Calculate a ranking, 1: less likely to suscribe; 5: the most likely to suscribe
    ranking = np.ceil(y_prob_pred/0.2) # lead suscription ranking from 1 to 5, 5 is the most likely
    ranking = ranking.astype('int')

    # Store predictions in a Dataframe and sort by subscription probability
    predictions = pd.DataFrame({
        "Subscribes": y_pred,
        "Probability_yes": y_prob_pred,
        "Ranking": ranking, 
        "Contacted:": False,
    })
    predictions.sort_values('Probability_yes', ascending = False, inplace = True)
    return predictions
def userapp(df):
    edited_df = st.data_editor(
    df,
    column_config={
        "Subscribes": "User may subscribe?",
        
        "Probability_yes": st.column_config.ProgressColumn(
            label= "Subscription probability",
            format=".0%",
            min_value=0,
            max_value=1),
                                                                  
        "Ranking": st.column_config.NumberColumn(
            label="Subscription likelihood ranking",
            help="1: Less likely to suscribe; 5: Most likely to suscribe",
            min_value=1,
            max_value=5,
            step=1,
            format="%d ⭐",
        ),
        "Contacted": st.column_config.CheckboxColumn(
            label="Contacted?",
            help="Check the box if the lead was called already",
            
            )
    },
    disabled=["Subscribes", "Probability_yes", "Ranking"],
    hide_index=False)
    return edited_df

if __name__=='__main__':
    # Run importing and cleaning
     # Import data 
    X_train, X_test, Y_train, Y_test = etl.opensplitdata()
    # Clean data
    X_test = cleandata(X_test)
    # Predict
    predictions = leadprediction(X_test)
    userapp(predictions)


