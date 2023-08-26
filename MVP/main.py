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


#### MAIN SCRIPT ####

# Install libraries
#pip install numpy sklearn imblearn shap catboost pathlib

#Import libraries
import numpy as np
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
#from sklearn.compose import ColumnTransformer
#from sklearn.base import BaseEstimator, TransformerMixin

# EDA & Visualization
import shap

# Modelling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn import metrics
from catboost import CatBoostClassifier

# Import defined variables
import settings

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
import pandas as pd

def readfile(file_to_open):
    bankleads = pd.read_csv(file_to_open)
    return bankleads

bankleads = readfile(file_to_open)

#Prepare data
X = bankleads.iloc[:,:-1]
Y = bankleads['y']

# Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

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

# ## 3. Model training, selection and evaluation
# ### 3.1. Models with oversampled data and all variables
# ##### i. Oversampling

# Oversample train data
oversample = SMOTE(random_state=11)
X_train_sm, Y_train_sm = oversample.fit_resample(X_train, Y_train)

# #### Catboost with oversampling
cbc = CatBoostClassifier(random_seed=42, logging_level='Silent') # Do not output any logging information.

# Train model 
cat_features = categorical_selector(X_train_sm, num_cols)
cbclf = cbc.fit(X_train_sm,Y_train_sm, 
        cat_features = cat_features,  
        eval_set=(X_test, Y_test) 
       )
# Predict values
y_pred = cbclf.predict(X_test)
# Predict probabilities
y_prob_pred = cbclf.predict_proba(X_test)[:,1]

# Store predictions in a Dataframe and sort by subscription probability
predictions = pd.DataFrame({
    "Subscribes": y_pred,
    "Probability_yes": y_prob_pred
})

predictions.sort_values('Probability_yes', ascending = False, inplace = True)
predictions.head()

# Model evaluation:
    # Training data performance
    
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}

scores = cross_validate(cbclf, X_train_sm, Y_train_sm, cv=5, scoring=scoring)

print ('Accuracy score : %.3f' % scores['test_acc'].mean())
print ('Precision score : %.3f' % scores['test_prec_macro'].mean())
print ('Recall (sensitivity) score : %.3f' % scores['test_rec_macro'].mean())
print ('F1 score : %.3f' % scores['test_f1_macro'].mean())

# Test data performance
print(metrics.classification_report(Y_test, y_pred, target_names=tagsy, digits=3))

# ### 3.3. Feature importance
# #### b. Based on SHAP 
explainer = shap.TreeExplainer(cbclf)
shapvalues = explainer.shap_values(X_test)
shap.summary_plot(shapvalues, X_test, plot_type='bar')

#  ### 3.4. Customer profile and model explainability
shap.summary_plot(shapvalues, X_test)

