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
import pandas as pd
import numpy as np
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


# Import user defined variables
import usersetts as setts

# Import preprocessing
import etl

#from sklearn.compose import ColumnTransformer
#from sklearn.base import BaseEstimator, TransformerMixin

# EDA & Visualization
import shap

# Modelling
from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import CatBoostClassifier


if __name__=='__main__':
    # 1. Data importing 
    file_to_open = setts.filepath()
    X_train, X_test, Y_train, Y_test = etl.opensplitdata(file_to_open, setts.target_var)
    
    # 2. Data processing 
    


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

