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


#### ETL SCRIPT ####

#import libraries

# Basic libraries
import usersetts as setts # user settings
import pandas as pd
import numpy as np
from pickle import dump, load

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ## 1. Data importing 
# Define function to read a csv into a Dataframe and then split it into train and test datasets

def opensplitdata(file_to_open = None, target_var = setts.target_var):
    if file_to_open is None:
         file_to_open = setts.filepath()
    data = pd.read_csv(file_to_open)
    #Prepare data
    X = data.loc[:, data.columns != target_var]
    Y = data[target_var]
    # Split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return X_train, X_test, Y_train, Y_test

# ## 2. Data processing 

# Define categorical variables
def categorical_selector(df, num_cols = setts.num_cols):
    cat_features =[ele for ele in df.columns.tolist() if ele not in num_cols]
    return cat_features

# Most classification models do not admit string variables as inputs, therefore, they must be converted to numeric types. Also, ordinal variables should be encoded accordingly in order to be interpreted correctly by the classification models. Next we will perform some operations on data to overcome this issues.

# ### 2.1. Ordinal features
# Create a copy of ordinal columns
def ordinalcopier(df, cols = setts.ordinal_columns):
    """
    Takes a dataframe and returns a copy of it. By default, takes ordinal columns. 
    
    Parameters:
    
    data: A dataframe
    cols: list of columns. By default, takes user specified ordinal columns. If not specified, takes all columns.
    """
    if cols is None:
        cols = df.columns.tolist()
    
    df[cols] = df[cols].copy()
    return df

def ord_imputer(df, cols = setts.ordinal_columns, strategy='most_frequent', missing_values='unknown', save_imputer = False, load_imputer = False):
    """Function that imputes data using sklearn's SimpleImputer
    
    Parameters:
        df: Dataframe
        cols: List of columns
        strategy: imputation strategy. By default, is set to 'most_frequent'
        missing_values: Placeholder for the missing values. By default, treats the word 'unknown' as missing.   
        save_imputer: boolean, True: serializes sklearn imputer, False: do not save sklearn imputer and checks if a pickle object is to be loaded.
        load_imputer: boolean, True: loads serialized sklearn imputer. False: do not load imputer. 
    For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html 
    """
   
    # Optional: Serialize imputer
    if save_imputer is True:
        ordimputer = SimpleImputer(strategy = strategy, missing_values = missing_values)
        dump(ordimputer, open('ordinalimputer.pkl','wb'))
        df[cols] = ordimputer.fit_transform(df[cols])
    elif load_imputer is True:
        ordimputer = load(open('ordinalimputer.pkl','rb'))
        df[cols] = ordimputer.fit_transform(df[cols])
    else:
        ordimputer = SimpleImputer(strategy = strategy, missing_values = missing_values)
        df[cols] = ordimputer.fit_transform(df[cols])
              
    return df

def ord_encode(df, cols = setts.ordinal_columns, categories=setts.ordinal_categories, handle_unknown='use_encoded_value', unknown_value=np.nan, save_encoder = False):
    """Function that encodes ordinal variables using sklearn's OrdinalEncoder

    Parameters:
        df: dataframe
        cols: list of ordinal columns
        categories (list): _description_. Defaults to setts.ordinal_categories.
        handle_unknown (str): _description_. Defaults to 'use_encoded_value'.
        unknown_value (int or np.nan): _description_. Defaults to np.nan.
    """
    ordencoder = OrdinalEncoder(categories=categories, handle_unknown=handle_unknown, unknown_value=unknown_value)
    # Optional: Serialize encoder
    if save_encoder is True:
        dump(ordencoder, open('ordinalencoder.pkl','wb'))
    else:
        pass
    # Encode ordinal variables
    df[cols] = ordencoder.fit_transform(df[cols])
    df[cols] = df[cols].astype('int')
    return df

# ### 2.2. Nominal features

# Function to define a list of nominal columns
def nominal_cols(df, ordinal_cols = setts.ordinal_columns):
    """
    Returns a list of non numerical and non ordinal string variables.
    
    Parameters:
        df: A pandas DataFrame.
        ordinal_columns: list of the names of ordinal columns
    """
    nom_cols = df.select_dtypes(include='object').columns.tolist()
    for item in ordinal_cols:
        if item in nom_cols:
            nom_cols.remove(item)
        else:
            pass
    return nom_cols

# Define a function to encode string variables as dummies.

def one_hot_encode(df, nominal_columns = None):
    """
    Apply one hot encoding to user selected variables, then, drop one column if the column has a binary category.
  
  Args:
    df: A pandas DataFrame.
    ordinal_columns: list of the names of nominal columns
    
  Returns:
    A pandas DataFrame with the encoded variables.
    """       
    # One hot encode the string type variables.
    if nominal_columns is None:
        nominal_columns = nominal_cols(df)
    df = pd.get_dummies(df, drop_first=True, dtype=int, columns = nominal_columns)
    
    # Return the encoded DataFrame.
    return df

def nom_imputer(df, nom_cols = None, ordinal_cols = setts.ordinal_columns, strategy='most_frequent', missing_values='unknown', save_imputer = False, load_imputer = False):
    """Function that imputes data using sklearn's SimpleImputer
    
    Parameters:
        df: Dataframe
        cols: List of columns
        strategy: imputation strategy. By default, is set to 'most_frequent'
        missing_values: Placeholder for the missing values. By default, treats the word 'unknown' as missing.   
        
    For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html 
    """
    if nom_cols is None:
        nom_cols = nominal_cols(df, ordinal_cols)

    # Optional: Serialize imputer
    if save_imputer is True:
        nomimputer = SimpleImputer(strategy = strategy, missing_values = missing_values)    
        dump(nomimputer, open('nominalimputer.pkl','wb'))
        df[nom_cols] = nomimputer.fit_transform(df[nom_cols])
    elif load_imputer is True:
        nomimputer = load(open('nominalimputer.pkl','rb'))
        df[nom_cols] = nomimputer.fit_transform(df[nom_cols])
    else:
        nomimputer = SimpleImputer(strategy = strategy, missing_values = missing_values)
        df[nom_cols] = nomimputer.fit_transform(df[nom_cols])
    return df 

# ### 2.3. Numeric features
# Rescale numeric features in order to prepare them for resampling
def scale_numeric_columns(data, columns = setts.num_cols):
    scaler = MinMaxScaler()
    scaled_data = data.copy()
    scaled_data[columns] = scaler.fit_transform(scaled_data[columns])
    return scaled_data