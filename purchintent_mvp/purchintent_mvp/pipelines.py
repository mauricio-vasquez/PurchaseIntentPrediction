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

#### DATA PROCESSING SKLEARN PIPELINES SCRIPT ####

#import libraries
import pandas as pd
# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Import modules
import usersetts as setts
import etl

# 1. Create the pipeline for ordinal variables

# Create pipeline for ordinal variables 
ord_pipe = Pipeline( [ 
    ('copyordinal', FunctionTransformer(etl.ordinalcopier)), 
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values='unknown')),
    ('encoder', OrdinalEncoder(categories=setts.ordinal_categories, handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ] )

