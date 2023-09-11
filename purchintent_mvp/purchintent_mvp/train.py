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

# EDA & Visualization
import shap

# Modelling
from sklearn import metrics
from catboost import CatBoostClassifier
import etl

# ## 3. Model training, selection and evaluation
# ### 3.1. Models with oversampled data and all variables
# ##### i. Oversampling

# Oversample train data
oversample = SMOTE(random_state=11)
X_train_sm, Y_train_sm = oversample.fit_resample(X_train, Y_train)

# #### Catboost with oversampling
cbc = CatBoostClassifier(random_seed=42, logging_level='Silent') # Do not output any logging information.

# Train model 
cat_features = etl.categorical_selector(X_train_sm, num_cols)
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