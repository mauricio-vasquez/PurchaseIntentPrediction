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

# Create indepent transformers for ordinal variables
def functiontransformer(object):
    transformed_obj = FunctionTransformer(object)
    return transformed_obj

ordcopytransform = functiontransformer(etl.ordinalcopier)

ord_pipe = Pipeline( [ 
    ('copyordinal', ordcopytransform), 
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values='unknown')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ] )

"""
# Ordinal variables transform
X_train[setts.ordinal_columns] = pipelines.ord_pipe.fit_transform(X_train[setts.ordinal_columns])
X_test[setts.ordinal_columns] = pipelines.ord_pipe.fit_transform(X_test[ordinal_columns])

X_train[setts.ordinal_columns] = X_train[setts.ordinal_columns].astype('int')
X_test[setts.ordinal_columns] = X_test[setts.ordinal_columns].astype('int')

# 2. Create the pipeline for nominal variables
"""

