import importlib.resources as pkg_resources
import os

import mne

import os
print(os.getcwd())
from alice_ml import data
from alice_ml.features import get_features_from_mne


def test_get_features_from_mne():
    
    raw = mne.io.read_raw_fif(pkg_resources.open_binary(data, 'test_raw.fif'), preload=True, verbose=False)
    
    # TODO: stick to alice_ml path
    path_to_ica = os.path.join(os.getcwd(), 'alice_ml', 'data', 'test_ica.fif')
    ica = mne.preprocessing.read_ica(path_to_ica)

    feature_df = get_features_from_mne(raw, ica)

    # 15 ics in the test sample
    assert feature_df.shape[0] == 15
    # 11 main features
    assert feature_df.shape[1] == 11