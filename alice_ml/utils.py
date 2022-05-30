import numpy as np
import pandas as pd
import mne


# TODO Import from existing project's vis.py
def get_epochs_from_df(ica_data, sfreq):
    epochs = ica_data.index
    n_epochs = len(epochs)
    counts = ica_data.apply(len)
    n_samples = counts[epochs[0]]
    assert np.all(counts == n_samples)

    np_data = np.empty((n_epochs, 1, n_samples))

    for idx, epoch in enumerate(epochs):
        np_data[idx, 0, :] = ica_data[epoch]

    info = mne.io.meas_info.create_info(['ICA000'], sfreq=sfreq, ch_types="misc", verbose=False)

    epochs_from_df = mne.EpochsArray(np_data, info, verbose=False)

    return epochs_from_df


def trim(series, q):
    return series[series < series.quantile(q)]