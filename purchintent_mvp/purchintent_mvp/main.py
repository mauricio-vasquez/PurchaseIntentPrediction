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


#### MAIN SCRIPT ####

# Install libraries
#pip install numpy sklearn imblearn shap catboost pathlib

#Import libraries
import pandas as pd
import numpy as np
import usersetts as setts

# Import preprocessing
import etl

# Import training
import train

def cleandata(df):  
    # Data processing 
    ### Ordinal features
    df = etl.ordinalcopier(df)   
    df = etl.ord_imputer(df, save_imputer = True)
    df = etl.ord_encode(df, save_encoder = True)

    ### Nominal features
    df = etl.nom_imputer(df, save_imputer = True)
    df = etl.one_hot_encode(df)

    ### Numeric features
    df = etl.scale_numeric_columns(df)
    return df

def training(X_train,Y_train, X_test, Y_test):
    train.catboostcl(X_train,Y_train, X_test, Y_test)


if __name__=='__main__':
    # Run importing and cleaning
     # Import data 
    X_train, X_test, Y_train, Y_test = etl.opensplitdata()
    # Clean data
    X_train = cleandata(X_train)
    X_test = cleandata(X_test)
    # Train model
    cbclf = training(X_train,Y_train, X_test, Y_test)


