# # APZIVA AI RESIDENCY PROGRAM
# 
# ## Project 2: Intent Marketing - MINIMUM VIABLE PRODUCT ##
# ### Lead prediction system for the banking business

# ### Prepared by: Mauricio Vásquez A. 
# ### Mentor: Swarnabha Ghosh. 
# ### Last updated on: August 2023
# 
# #### Contact email: mauricio_vasquez_andrade@hotmail.com
# #### LinkedIn: https://www.linkedin.com/in/mauricio-vasquez-andrade-ecuador/


#### ETL SCRIPT ####

#import libraries
import pandas as pd
# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Function to define a list of nominal columns
def nominal_selector(df, ordinal_cols):
    """
    Returns a list of non numerical and non ordinal string variables.
    
    Args:
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

# Define categorical variables
def categorical_selector(df, num_cols):
    cat_features =[ele for ele in df.columns.tolist() if ele not in num_cols]
    return cat_features

# ## 1. Data importing 
# Define function to read a csv into a Dataframe

def opensplitdata(file_to_open, target_var):
    bankleads = pd.read_csv(file_to_open)
    #Prepare data
    X = bankleads.loc[:, bankleads.columns != target_var]
    Y = bankleads[target_var]
    # Split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return X_train, X_test, Y_train, Y_test

# ## 2. Data processing 

# Most classification models do not admit string variables as inputs, therefore, they must be converted to numeric types. Also, ordinal variables should be encoded accordingly in order to be interpreted correctly by the classification models. Next we will perform some operations on data to overcome this issues.
# ### 2.1. Ordinal features

# Create a copy of ordinal columns
def ordinalcopier(data, cols = ordinal_columns):
    """
    Takes a dataframe and returns a copy of it. By default, takes all dataframe columns, unless specified. 
    
    Parameters:
    
    data: A dataframe
    cols = selected columns. By default, takes all dataframe columns
    """
    if cols is None:
        cols = data.columns.tolist()
    
    data[cols] = data[cols].copy()
    return data

# Create indepent transformers for ordinal variables
ordcopytransform = FunctionTransformer(ordinalcopier)

# Create the pipeline for ordinal variables
ord_steps = ( [ 
    ('copyordinal', ordcopytransform), 
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values='unknown')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ] )

ord_pipe = Pipeline(ord_steps)

X_train[ordinal_columns] = ord_pipe.fit_transform(X_train[ordinal_columns])

X_train[ordinal_columns] = X_train[ordinal_columns].astype('int')

X_test[ordinal_columns] = ord_pipe.fit_transform(X_test[ordinal_columns])

X_test[ordinal_columns] = X_test[ordinal_columns].astype('int')

# ### 2.2. Nominal features

# Select nominal columns
nom_cols = nominal_selector(X_train, ordinal_columns)

# Define a function to encode string variables as dummies.

def one_hot_encode(df, nominal_columns):
    """
    Apply one hot encoding to user selected variables, then, drop one column if the column has a binary category.
  
  Args:
    df: A pandas DataFrame.
    ordinal_columns: list of the names of nominal columns
    
  Returns:
    A pandas DataFrame with the encoded variables.
    """       
    # One hot encode the string type variables.
    encoded_df = pd.get_dummies(df, drop_first=True, dtype=int, columns = nominal_columns)
    
    # Return the encoded DataFrame.
    return encoded_df


# ##### a. Train dataset transformation

# Impute unknown values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent', missing_values='unknown')
X_train[nom_cols] = imputer.fit_transform(X_train[nom_cols])
X_train = one_hot_encode(X_train, nom_cols)


# ##### b. Test dataset transformation
# Impute unknown values with the most frequent value

imputer = SimpleImputer(strategy='most_frequent', missing_values='unknown')
X_test[nom_cols] = imputer.fit_transform(X_test[nom_cols])
X_test = one_hot_encode(X_test, nom_cols)

# ### 2.3. Numeric features
# Rescale numeric features in order to prepare them for resampling
def scale_numeric_columns(data, columns):
    scaler = MinMaxScaler()
    scaled_data = data.copy()
    scaled_data[columns] = scaler.fit_transform(scaled_data[columns])
    return scaled_data

X_train = scale_numeric_columns(X_train, num_cols)
X_test = scale_numeric_columns(X_test, num_cols)