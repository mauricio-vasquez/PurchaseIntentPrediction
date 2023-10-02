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

#### CONFIG SCRIPT ####
# ## Define config variables

# a. Paths
from pathlib import Path, PureWindowsPath
projectpath = Path("C:/Users/USER/Documents/Mauricio V/Apziva/Projects/Project 2 - IntentMarketing")
filedir = projectpath / 'Data' / 'Apziva'
packagedir = projectpath / 'PurchaseIntentPrediction' / 'purchintent_mvp' / 'purchintent_mvp'
pickledir = packagedir / 'pickle_objs'

file = 'term-deposit-marketing-2020.csv'

def filepath(file = file, path = filedir):
    file_to_open = Path(path) / file
    return file_to_open

# b. Data labels and columns
target_var = 'y'
tagsy = ['No (0)', 'Yes (1)']
ordinal_columns = ['education', 'month']
ordinal_categories = [['primary', 'secondary', 'tertiary'],
                      ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]

#Define numeric columns
num_cols = ['age','balance', 'duration', 'campaign']