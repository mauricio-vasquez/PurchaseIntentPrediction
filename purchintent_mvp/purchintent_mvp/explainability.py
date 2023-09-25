import shap

def featimportanceviz(df):
    # ### 3.3. Feature importance
    # #### b. Based on SHAP 
    model = load(open('catboostclassifier.pkl','rb'))
    explainer = shap.TreeExplainer(model)
    shapvalues = explainer.shap_values(df)
    return shap.summary_plot(shapvalues, X_test, plot_type='bar')