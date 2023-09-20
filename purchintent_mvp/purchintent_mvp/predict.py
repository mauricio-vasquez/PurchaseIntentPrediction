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
import usersetts as setts
from pickle import load

# Import preprocessing
import etl

def cleandata(df):  
    # Data processing 
    ### Ordinal features
    df = etl.ordinalcopier(df)   
    df = etl.ord_imputer(df, load_imputer = True)
    df = etl.ord_encode(df, load_encoder = True)

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

    # Store predictions in a Dataframe and sort by subscription probability
    predictions = pd.DataFrame({
        "Subscribes": y_pred,
        "Probability_yes": y_prob_pred
    })
    predictions.sort_values('Probability_yes', ascending = False, inplace = True)
    return predictions
    
############ 19sep2023 Me quedé aquí ###################

if __name__=='__main__':
    # Run importing and cleaning
     # Import data 
    X_train, X_test, Y_train, Y_test = etl.opensplitdata()
    # Clean data
    X_train = cleandata(X_train)
    X_test = cleandata(X_test)
    # Train model
    



"""
# ### 3.3. Feature importance
# #### b. Based on SHAP 
explainer = shap.TreeExplainer(cbclf)
shapvalues = explainer.shap_values(X_test)
shap.summary_plot(shapvalues, X_test, plot_type='bar')

#  ### 3.4. Customer profile and model explainability
shap.summary_plot(shapvalues, X_test)
"""