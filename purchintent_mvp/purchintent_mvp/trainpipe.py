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


#### MODEL TRAINING SCRIPT ####

# Modelling
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from pickle import dump
import etl

# ## 3. Model training, selection and evaluation
# ### 3.1. Models with oversampled data and all variables
# ##### i. Oversampling

def oversample(X_train,Y_train):
    # Oversample train data
    oversample = SMOTE(random_state=11)
    X_train_sm, Y_train_sm = oversample.fit_resample(X_train, Y_train)
    return X_train_sm, Y_train_sm

#### Catboost with oversampling
def catboostcl(X_train,Y_train, X_test, Y_test, cat_features = None):
    X_train_sm, Y_train_sm = oversample(X_train,Y_train)
    cbc = CatBoostClassifier(random_seed=42, logging_level='Silent') # Do not output any logging information. 
    # Train model 
    if cat_features is None:
        cat_features = etl.categorical_selector(X_train_sm)
    cbclf = cbc.fit(X_train_sm,Y_train_sm, 
            cat_features = cat_features,  
            eval_set=(X_test, Y_test) 
        )
    dump(cbclf, open('catboostclassifier.pkl','wb'))
    return cbclf