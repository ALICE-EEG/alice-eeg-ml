import importlib.resources as pkg_resources
from itertools import chain

import numpy as np
import pandas as pd
import mne
import statsmodels.api as sm
from scipy.signal import find_peaks, peak_prominences

from alice_ml.utils import trim
from alice_ml.preprocessing import IC
from alice_ml import data


eye_move_example = np.load(pkg_resources.open_binary(data, 'eye_move_example.npy'))
eye_blink_example = np.load(pkg_resources.open_binary(data, 'eye_blink_example.npy'))
mean_pca = np.load(pkg_resources.open_binary(data, 'pca_mean.npy'))
components_pca = np.load(pkg_resources.open_binary(data, 'pca_components.npy'))

def _cross_corr(epoch_eye, epoch):
    ccov = np.correlate(epoch_eye - epoch_eye.mean(), epoch - epoch.mean(), mode='same')
    ccor = ccov / (len(epoch_eye) * epoch_eye.std() * epoch.std())

    return ccor

def __alpha_peak_from_psd(freqs, psd):
    alphs_inds = np.where((freqs>=5)&(freqs<=15))[0]
    change = np.linspace(-1, 1, len(freqs))
    
    psd = (10 * np.log10(psd * 100/2))
    xs = sm.add_constant(1. / freqs)
    
    lr = sm.OLS(psd.T, xs).fit()
    residuals = lr.get_influence().resid_studentized_internal
    res_pk, _ = find_peaks(residuals[alphs_inds], height=1, 
                          prominence=0.3, width=2.7, rel_height=0.5)
    psd_pk, _ = find_peaks(psd[alphs_inds], height=0,
                           prominence=0.15, width=0, rel_height=0.5)
    
    res_pl, _ = find_peaks(residuals[alphs_inds], height=1, prominence=0.15,
                           width=0, rel_height=0.5)
    psd_pl, _ = find_peaks(psd[alphs_inds] + 10*change[alphs_inds], 
                           prominence=0.05, height=0, width=0, rel_height=0.5)
    
    def _to_freq(pk_list):
        return [freqs[alphs_inds][p] for p in pk_list]
    
    return _to_freq(res_pk), _to_freq(psd_pk), _to_freq(res_pl), _to_freq(psd_pl), lr


def compute_K(ic, mean_shift=False, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        mean_shift (bool, optional): if set to True, the average over epoch is subtracted. Defaults to False.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Temporal Kurtosis feature.
    """
    def _epoch_kurtosis(epoch):
        if not mean_shift:
            return np.mean(epoch ** 4) / (np.mean(epoch ** 2) ** 2) - 3
        return np.mean((epoch - epoch.mean()) ** 4) / (epoch.var() ** 2) - 3

    epochs = ic.signal.apply(_epoch_kurtosis)

    if not thres:
        return epochs.mean()
    return trim(epochs, thres).mean()


def compute_MEV(ic, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Maximum Epoch Variance feature.
    """
    vars = ic.signal.apply(np.var)

    if not thres:
        return vars.max() / vars.mean()
    return vars.quantile(thres) / trim(vars, thres).mean()


FA = {'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
      'af3', 'af4', 'af7', 'af8',
      'fp1', 'fpz', 'fp2','e27','e123','e33','e122','e32','e1','e23',
      'e3','e26','e2','e22','e15','e21','e14','e9'}
PA = {'cpz', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6',
      'pz', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 't5', 't6', 'p9', 'p10',
      'poz', 'po3', 'po4', 'po7', 'po8',
      'oz', 'o1', 'o2','e55','e37','e87','e42','e93','e47',
      'e98','e62','e60','e85','e52','e92','e51','e97',
      'e58','e96','e64','e95','e72','e67','e77','e65','e90',
      'e75','e70','e83'}
LE = {'f3', 'f5', 'f7', 'f9',
      'af7','e24','e27','e33','e32','e26'}
RE = {'f4', 'f6', 'f8', 'f10',
      'af6','e124','e123','e122','e1','e3'}

FC = {'fp1','fpz','fp2','f3','fz','f4','fc3','fcz','fc4','c3','cz','c4',
      'e22','e15','e21','e14','e9','e24','e11','e124',
      'e29','e6','e111','e36','e129','e104'}
FC_not={'f7','ft7','t3','tp7','t5','t6','tp8','t4','ft8','f8','cp3',
        'cpz','cp4','p3','pz','p4','o1','oz','o2',
        'e33','e34','e45','e46','e58','e96','e102','e108','e116',
        'e122','e42','e55','e93','e52','e62','e92','e70','e83','e75'}
FP= {'c3','cz','c4','cp3','cpz','cp4','p3','pz','p4','o1','oz','o2',
    'e36','e129','e104','e42','e55','e93','e52','e62','e92','e70','e83','e75'}
FP_not={'f7','ft7','t3','tp7','t5','t6','tp8','t4','ft8','f8','fp1','fpz',
        'fp2','f3','fz','f4','fc3','fcz','fc4',
        'e33','e34','e45','e46','e58','e96','e102','e108','e116',
        'e122','e22','e15','e21','e14','e9','e24','e11','e124','e29'}


def compute_AT(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Alpha Topography - Difference between weights in frontal-central sites and other sites
    """
    return np.abs(ic.select_weights(FC).mean()) - np.abs(ic.select_weights(FC_not).mean())

def compute_MT(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Mu Topography - Difference between weights in occipital-parietal sites and other sites
    """
    return np.abs(ic.select_weights(FP).mean()) - np.abs(ic.select_weights(FP_not).mean())

def compute_AMALB(ic)  -> float:
    
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Average Magnitude in ALpha Band
    """


    freqs, psd=ic.psd(verbose=False)
    
    alphs_inds=np.where((freqs>=6)&(freqs<=12))


    alphs_inds_not=np.where((freqs<6)|(freqs>12))
    
    mean_psd = psd.mean(axis=0)
    
    return np.mean(mean_psd[alphs_inds])/np.mean(mean_psd[alphs_inds_not])
    
    



def compute_SAD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Average Difference feature.
    """
    return np.abs(ic.select_weights(FA).mean()) - np.abs(ic.select_weights(PA).mean())


def compute_SVD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Variance Difference feature.
    """
    return ic.select_weights(FA).var() - ic.select_weights(PA).var()


def compute_SED(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Eye Difference feature.
    """
    return np.abs(ic.select_weights(LE).mean() - ic.select_weights(RE).mean())


def compute_MIF(ic):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Myogenic identification feature.
    """
    freqs, psd = ic.psd(verbose=False)
    mean_psd = psd.mean(axis=0)
    return mean_psd[freqs > 20].sum() / mean_psd.sum()


def compute_CORR_BL(ic, thres=0.65):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Correaltion with eye blink example
    """
    in_epoch_corrs = [_cross_corr(eye_blink_example, epoch) for epoch in ic.signal.values]
    pattern_probs = [np.mean(abs(epoch_corr) > thres) for epoch_corr in in_epoch_corrs]

    return np.mean(pattern_probs)


def compute_CORR_MOVE(ic, thres=0.65):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Correaltion with eye movement example
    """
    in_epoch_corrs = [_cross_corr(eye_move_example, epoch) for epoch in ic.signal.values]
    pattern_probs = [np.mean(abs(epoch_corr) > thres) for epoch_corr in in_epoch_corrs]

    return np.mean(pattern_probs)


def compute_CIF(ic):
    # TODO Implement feature. Address low frequency resolution
    raise NotImplementedError


def compute_alpha_features(ic, average_epochs=False):
    peaks_data = pd.DataFrame()
    freqs, psd = ic.psd(verbose=False)

    freq_thr = np.where((freqs>=1)&(freqs <= 50))[0]
    freqs = freqs[freq_thr]
    psd = psd[:, freq_thr] 

    epochs_with_pk = 0
    epochs_with_pl = 0
    peaks_coords = []
    plats_coords = []
    
    if average_epochs:
        psd = psd.mean(axis=0).reshape(1, -1)
    
    for i in range(psd.shape[0]):
        res_pk, psd_pk, res_pl, psd_pl, _ = __alpha_peak_from_psd(freqs, psd[i, :])

        if len(res_pk) > 0 and len(psd_pk) > 0:
            epochs_with_pk += 1
            peaks_coords = peaks_coords + res_pk

        if len(res_pl)>0 and len(psd_pl)>0:
            epochs_with_pl += 1
            plats_coords = plats_coords + res_pl
            
    epochs_with_pk = epochs_with_pk / psd.shape[0]
    epochs_with_pl = epochs_with_pl / psd.shape[0]

    mean_peak_freq = np.mean(peaks_coords) if len(peaks_coords) > 0 else None
    mean_plat_freq = np.mean(plats_coords) if len(plats_coords) > 0 else None
    
    return epochs_with_pk, epochs_with_pl, mean_peak_freq, mean_plat_freq


default_features = {'K': compute_K,
                    'MEV': compute_MEV,
                    'SAD': compute_SAD,
                    'SVD': compute_SVD,
                    'SED': compute_SED,
                    'MIF': compute_MIF,
                    'CORR_BL': compute_CORR_BL,
                    'CORR_MOVE': compute_CORR_MOVE,
                    'AT':compute_AT,
                    'MT':compute_MT,
                    'AMALB':compute_AMALB}


def build_alice_features_df(data, default=True, custom_features={}):
    """
    Computes the feature matrix for the dataset of components.

    Args:
        data (dataset): A mapping of {ic_id: IC}. Compatible with the dataset representaion produced by load_dataset().
        default (bool, optional): Determines wether to compute a standard selection of features for the dataset. Defaults to True.
        custom_features (dict, optional): A mapping of custom features that will be computed for the dataset.
        The format is {feature_name: compute_feature} where compute_feature is a function with the only argument IC. Defaults to {}.

    Returns:
        pd.Dataframe: The feature matrix for the dataset.
    """
    def get_iter():
        if default:
            return default_features.items()
        else:
            return chain(default_features.items(), custom_features.items())

    features = [feature_name for feature_name, _ in get_iter()]
    idx = []
    rows = []
    for ic_id, ic in data.items():
        row = []
        idx.append(ic_id)
        for feature_name, compute_feature in get_iter():
            row.append(compute_feature(ic))
        rows.append(row)
    alice_feature_df = pd.DataFrame(rows, index=idx, columns=features)

    return alice_feature_df


def build_alpha_features_df(data, average_epochs=False):
    """
    Computes the feature matrix for the dataset of components.

    Args:
        data (dataset): A mapping of {ic_id: IC}. Compatible with the dataset representaion produced by load_dataset().
        average_epochs (bool, optional): . Defaults to False.

    Returns:
        pd.Dataframe: The feature matrix for the dataset.
    """

    features = ['A_peaks', 'A_plateaus', 'A_peaks_freq', 'A_plateaus_freq']
    idx = []
    rows = []
    for ic_id, ic in data.items():
        row = []
        idx.append(ic_id)
        row = compute_alpha_features(ic)
        rows.append(row)
    alpha_feature_df = pd.DataFrame(rows, index=idx, columns=features)

    return alpha_feature_df

def build_pca_features_df(data):
    """
    Computes the feature matrix for the dataset of components.

    Args:
        data (dataset): A mapping of {ic_id: IC}. Compatible with the dataset representaion produced by load_dataset().
        average_epochs (bool, optional): . Defaults to False.

    Returns:
        pd.Dataframe: The feature matrix for the dataset.
    """
    features = ['pca1', 'pca2']
    idx = []
    rows = []
    channels =  ['fp1','fp2','f7','f3','fz','f4','f8','t3','c3','cz','c4','t4','t5','p3','pz','p4','t6','o1','o2']  
    channels_ = ['fp1','fp2','f7','f3','fz','f4','f8','t7','c3','cz','c4','t8','p7','p3','pz','p4','p8','o1','o2'] # for EEG system with 64 channels
    for ic_id, ic in data.items():
        row = []
        idx.append(ic_id)
        try:
            row = (ic.weights[channels].values - mean_pca) @ components_pca.T
        except:
            try:
                row = (ic.weights[channels_].values- mean_pca) @ components_pca.T
            except: # some channels have zero weights in IC, so 'ic.weights[channels_/channels]' gives errors 
                ch_weights = []
                for ch in channels:
                    if ch in ic.weights.index:
                        ch_weights.append(ic.weights[ch])
                    else:
                        ch_weights.append(0)
                row = (np.array(ch_weights) - mean_pca) @ components_pca.T
        rows.append(row)
    pca_feature_df = pd.DataFrame(rows, index=idx, columns=features)

    return pca_feature_df
      
def build_feature_df(data, average_epochs=False):
    """
    Computes the feature matrix for the dataset of components.

    Args:
        data (dataset): A mapping of {ic_id: IC}. Compatible with the dataset representaion produced by load_dataset().
        average_epochs (bool, optional): . Defaults to False.

    Returns:
        pd.Dataframe: The feature matrix for the dataset.
    """
    feature_df = pd.concat([build_alice_features_df(data),build_alpha_features_df(data),build_pca_features_df(data)], axis=1)

    return feature_df

def get_features_from_mne(obj, ica_obj, features = 'all'):
    ica_df = ica_obj.get_sources(obj).to_data_frame()
    if isinstance(obj, mne.io.BaseRaw):
        times = np.arange(ica_df.shape[0]) / ica_obj.info['sfreq']
        # get 2 seconds pseudo epochs
        pseudo_epoch_idx = [int(t / 2) for t in times]
        ica_df['epoch'] = pseudo_epoch_idx
        
        # crop last epoch if it is shorter than others
        first_epoch_len = len([idx for idx in pseudo_epoch_idx if idx == 0])
        last_epoch_len = len([idx for idx in pseudo_epoch_idx if idx == pseudo_epoch_idx[-1]])
        if first_epoch_len != last_epoch_len:
            ica_df = ica_df[ica_df['epoch'] != pseudo_epoch_idx[-1]]
    
    channels_to_use = [ch.lower() for ch in ica_obj.info['ch_names']]
    
    ic_names = [col for col in ica_df.columns if col not in ('epoch', 'time', 'condition')]

    data = {}

    for ic_idx, ic_name in enumerate(ic_names):
        df_weights = (
            pd.DataFrame({'ch_name': channels_to_use, 
                          'value': ica_obj.get_components()[:, ic_idx]})
            .set_index('ch_name')
            ['value'].rename('weights'))

        df_data = (
            ica_df[['epoch', ic_name]].rename(columns={ic_name: 'value'})
            .groupby('epoch')['value'].apply(np.array).rename('signal'))

        data[ic_name] = IC(ica_obj.info['sfreq'], signal=df_data, weights=df_weights)
        
    if features == 'all':
         features_df = build_feature_df(data)
    elif features == 'alice':
        features_df = build_alice_features_df(data)
    return features_df
