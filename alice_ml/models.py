import importlib.resources as pkg_resources
import joblib

import pandas as pd

from alice_ml.features import get_features_from_mne
from alice_ml import pretrained


def predict_mne(raw, ica, model = 'lr', flags = 'all'):
   
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

    scaler = joblib.load(pkg_resources.open_binary(pretrained, 'scaler.joblib'))

    models = {}
    if flags == 'all':
        flags = ['flag_brain', 'flag_muscles', 'flag_eyes']
    for flag in flags:
        models[flag] = joblib.load(pkg_resources.open_binary(pretrained, f'{model}_' + flag + '.joblib'))
    
    X = scaler.transform(features_df)
    cols = {}
    for flag in flags:
        cols[flag] = models[flag].predict_proba(X)[:, 1]
    
    pred_df = pd.DataFrame(cols, index=features_df.index)
    return pred_df
    
