import numpy as np
import mne
import pandas as pd


def calc_MI(X,Y,bins=100):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H


def mean_pairwise_MI(ica,data,bins=100):
    ica_df = ica.get_sources(data).to_data_frame()
    ic_names = [col for col in ica_df.columns if col not in ('epoch', 'time', 'condition')]
    
    signal = ica_df[ic_names].values.T
    M = np.zeros((signal.shape[0],signal.shape[0]))
    for i in range(signal.shape[0]):
        for j in range(signal.shape[0]):
            M[i,j] = calc_MI(signal[i],signal[j],bins)
    upper_mean = []
    for i in range(len(M)):
        upper_mean.extend(M[i][i+1:])
    return np.array(upper_mean).mean()