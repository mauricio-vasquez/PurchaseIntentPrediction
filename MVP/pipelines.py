#### Pipelines SCRIPT ####

#import libraries
import pandas as pd
# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

import etl

# 1. Create the pipeline for ordinal variables
ord_pipe = Pipeline( [ 
    ('copyordinal', etl.ordcopytransform), 
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values='unknown')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ] )

# 2. Create the pipeline for nominal variables

########## 29aug2023: Last edit here ############### 