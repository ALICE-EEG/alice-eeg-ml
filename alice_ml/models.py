import importlib.resources as pkg_resources
import joblib

import pandas as pd
import numpy as np

from alice_ml.features import get_features_from_mne
from alice_ml import pretrained
from sklearn_pmml_model.ensemble import PMMLGradientBoostingClassifier 
from sklearn_pmml_model.linear_model import PMMLLogisticRegression


def predict_mne(raw, ica, model = 'LR', features = 'all', flags = 'all'):
   
    """
    Using of pretrained model on a raw data to predict ICA label via machine learning techniques
    ----------
    raw : mne.io.Raw 
    
    ica : mne.preprocessing.ica.ICA
       
    model : str 
    What model to choose to predict ICA (currently available "lr" - for Logistic Regression, "xgb" - for XGBoost classifier,
    "svc" - for Support Vector Classifier)
    
    flags : str or list of flags
    What flags to predict, by default "all" - predicting all available flags ('flag_brain', 'flag_alpha', 'flag_mu', 'flag_muscles', 'flag_eyes', 'flag_heart', 'flag_ch_noise')
    """
    
    features_df = get_features_from_mne(raw, ica)

    models = {}
    ideces = {}
    if flags == 'all':
        flags = ['flag_mu', 'flag_alpha', 'flag_eyes']
    for flag in flags:
        if model == 'LR':
            models[flag] = PMMLLogisticRegression(pmml=pkg_resources.open_binary(pretrained, f"{model}_{flag}.pmml"))
            ideces[flag] = np.arange(17)
        elif model == 'XGB':
            models[flag] = PMMLGradientBoostingClassifier(pmml=pkg_resources.open_binary(pretrained, f"{model}_{flag}.pmml"))
            ideces[flag] = np.array([int(x[1:])-1 for x in list(models[flag].fields.keys())[1:]])
            
    scalers = {}
    for flag in flags:
        if flag=='flag_eyes':
            scalers[flag] = joblib.load(pkg_resources.open_binary(pretrained, 'scaler_eyes.joblib'))
        else:
            scalers[flag] = joblib.load(pkg_resources.open_binary(pretrained, 'scaler_mu_alpha.joblib'))
    
    cols = {}
    for flag in flags:
        X = scalers[flag].transform(features_df)
        cols[flag] = models[flag].predict_proba(X[:,ideces[flag]])[:, 1]
    
    pred_df = pd.DataFrame(cols, index=features_df.index)
    return pred_df
    
