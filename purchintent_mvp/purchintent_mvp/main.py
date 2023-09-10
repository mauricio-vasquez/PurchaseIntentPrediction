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

import usersetts as setts

# Import preprocessing
import etl

#from sklearn.compose import ColumnTransformer
#from sklearn.base import BaseEstimator, TransformerMixin

if __name__=='__main__':
    # 1. Data importing 
    X_train, X_test, Y_train, Y_test = etl.opensplitdata()
    
    # 2. Data processing 
    




