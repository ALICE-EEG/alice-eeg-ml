import importlib.resources as pkg_resources

import pandas as pd
import numpy as np
import mne

from alice_ml.models import predict_mne


def test_predict_mne():
    test_raw_path = 'alice_ml/data/test_raw.fif'
    test_ica_path = 'alice_ml/data/test_ica.fif'

    raw = mne.io.read_raw_fif(test_raw_path, preload=True)
    ica = mne.preprocessing.read_ica(test_ica_path)
    res = predict_mne(raw, ica, model='lr', flags='all')
    
    # check default models
    assert set(res.columns) == set(['flag_brain', 'flag_muscles', 'flag_eyes'])

    # check that result contains only probabilities
    assert (res < 0).sum().sum() == 0
    assert (res > 1).sum().sum() == 0