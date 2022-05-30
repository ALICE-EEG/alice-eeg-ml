from functools import lru_cache
from os.path import join
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alice_ml.utils import get_epochs_from_df

class IC:
    """
    A wrapper that represents the independent component. Contains the signal, weights of channels and the sampling frequency.
    """

    def __init__(self, freq, signal=None, weights=None, signal_path=None, weights_path=None):
        """
        If signal is None, signal_path must be set. If weights is None, weights_path must be set
        Setting signal_path and weights_path allows to use dynamic data loading with lru_cache. It is useful when your dataset is large.

        Args
            freq: Sampling frequency 
        """
        if (signal is None and signal_path is None):
            raise ValueError('signal or signal_path must be provided')
        if (weights is None and weights_path is None):
            raise ValueError('signal or signal_path must be provided')
        self._signal = signal
        self._weights = weights
        self._weights_path = weights_path
        self._signal_path = signal_path
        self.freq = freq
    
    @property
    @lru_cache(maxsize=10)
    def weights(self):
        if self._weights is None:
            return self._read_weights()
        return self._weights
    
    @property
    @lru_cache(maxsize=10)
    def signal(self):
        if self._signal is None:
            return self._read_signal()
        return self._signal
    
    def select_weights(self, channels):
        return self.weights[self.weights.index.isin(channels)]

    @lru_cache(maxsize=10)
    def psd(self, **kwargs):
        epochs = get_epochs_from_df(self.signal, self.freq)
        powers, freqs = mne.time_frequency.psd_multitaper(epochs, picks=[0], **kwargs)
        return freqs, powers.mean(axis=1)

    def plot_psd(self, returns=False):
        fig = plt.figure()

        freqs, powers = self.psd(verbose=False)
        plt.fill_between(freqs, powers.mean(axis=0) - powers.std(axis=0), powers.mean(axis=0) + powers.std(axis=0), alpha=0.2)
        plt.semilogy(freqs, powers.mean(axis=0))

        if returns:
            return fig

    def plot_topomap(self, returns=False):
        fig, ax = plt.subplots()

        outlines = 'head'

        res = 64
        contours = 6
        sensors = True
        image_interp = 'bilinear'
        show = True
        extrapolate = 'box'

        border = 0

        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        ten_twenty_montage_channels = {ch.lower(): ch for ch in ten_twenty_montage.ch_names}

        # get channels in format of ten_twenty_montage in right order

        channels_to_use_ = [ten_twenty_montage_channels[ch] for ch in self.weights.index]

        # create Info object to store info
        info = mne.io.meas_info.create_info(channels_to_use_, sfreq=256, ch_types="eeg")

        # using temporary RawArray to apply mongage to info
        mne.io.RawArray(np.zeros((len(channels_to_use_), 1)), info, copy=None, verbose=False).set_montage(ten_twenty_montage)

        # pick channels
        channels_to_use_ = [ch for ch in info.ch_names if ch.lower() in self.weights.index]
        info.pick_channels(channels_to_use_)

        _, pos, _, names, _, sphere, clip_origin = mne.viz.topomap._prepare_topomap_plot(info, 'eeg')

        outlines = mne.viz.topomap._make_head_outlines(sphere, pos, outlines, clip_origin)

        mne.viz.topomap.plot_topomap(
            self.weights, pos, res=res,
            outlines=outlines, contours=contours, sensors=sensors,
            image_interp=image_interp, show=show, extrapolate=extrapolate,
            sphere=sphere, border=border, axes=ax, names=names
        )

        if returns:
            return fig
    
    def _read_weights(self):
        if self._weights_path is None:
            raise RuntimeError('weights_path is None')
        return pd.read_csv(self._weights_path, index_col='ch_name')['value'].rename('weights')

    def _read_signal(self):
        if self._signal_path is None:
            raise RuntimeError('weights_path is None')
        return pd.read_csv(self._signal_path).groupby('epoch')['value'].apply(np.array).rename('signal')


def read_ic(dir, ics, ic_id, preload=True):
    path = Path(dir)
    signal_path = join(path, f'{ic_id}_data.csv')
    weights_path = join(path, f'{ic_id}_weights.csv')
    freq = ics.loc[ics['ic_id'] == ic_id, 'sfreq']
    if preload is True:
        signal = pd.read_csv(signal_path).groupby('epoch')['value'].apply(np.array).rename('signal')
        weights = pd.read_csv(weights_path, index_col='ch_name')['value'].rename('weights')
        return IC(freq, signal=signal, weights=weights)
    else:
        return IC(freq, signal_path=signal_path, weights_path=weights_path)
        


def load_dataset(dir='data', preload=True):
    path = Path(dir)
    ics = pd.read_csv(path/'ics.csv')
    ic_ids = list(ics['ic_id'])
    data = {ic_id: read_ic(dir, ics, ic_id, preload=preload) for ic_id in ic_ids}
    annotations = pd.read_csv(path/'annotations_raw.csv')
    return data, annotations


def get_flag_names(annotations) -> list:
    return list(annotations.columns[annotations.columns.str.startswith('flag_')])


def build_target_values(annotations, weights='equal'):
    assert weights in {'equal', 'uniform'}, 'Unknown weight type.'

    values = annotations[get_flag_names(annotations)].astype(float)

    if weights == 'uniform':
        values = values.div(values.sum(axis=1), axis='index')
    return pd.concat([annotations['ic_id'], values], axis=1)


def build_target_df(annotations, flags=None, strategy='majority', weights='equal', threshold=None) -> pd.DataFrame:
    """
    Args:

    annotations (pd.DataFrame):
        Dataframe containing targets.

    flags (list, optional):
        The list of flags for which to generate target. If set to None, all flags in annotaions will be selected.

    strategy (str, optional):
        The type of target value aggregation.
        There binary and non-binary strategies.
        Binary strategies are 'majority', 'any', 'all'.
        Non-binary strategies are 'mean', 'max', 'min'.
        Default strategy is 'majority'.

    weights (str, optional):
        The type of weight aggregation.
        When set to 'equal' all psoitive labels will have weight 1.
        When set to 'uniform' all psoitive labels will have weight  1 / n,
        where n is the number of positive labels assigned to this ic by this expert.
        Defaults to 'equal'.

    threshold (list or float, optional):
        Either the universal threshold value or a list of threshold values for each flag in `flags`.
        If provided, the output will be binary. Requires non-binary strategy.
        Defaults to None.

    Returns:

        pd.DataFrame: dataframe containing targets.
    """
    assert strategy in {'min', 'max', 'mean', 'all', 'any', 'majority'}, 'Unknown strategy.'

    if flags is None:
        flags = get_flag_names(annotations)

    target_values = build_target_values(annotations, weights)
    target_groups = target_values.groupby('ic_id')[flags]

    if strategy == 'majority':
        targets = target_groups.mean()
        return targets >= 0.5

    targets = getattr(target_groups, strategy)()

    if threshold is not None:
        assert strategy in {'min', 'max', 'mean'}, 'When provided threshold use non-binary strategy.'
        return targets >= threshold

    return targets

# Legacy
def get_target_temp(annotations, flag, threshold=0.5, agg_type='all_ones') -> pd.DataFrame:
    
    """
    Args:
        annotations (pd.DataFrame): dataframe containing targets.
        
        
        flags (dict, list, optional): either list of flag names for which to construct labels or a mapping {flag_name: threshold}. Then it will select each component with its own threshold value.
        If set to None, all flags in annotaions will be selected.
        
        
        agg_type (string): the principle of experts marks aggregation
            all_ones: equal union of all experts marks
            intercept_ones: only overlap of marks is accounted as correct
            weigths_of_ones: all marks are estimated according to their probabilty among experts
            weights: all marks are estimated  according to their probabilty among other marks of each components and among experts
            weigths_with_desicion: marks are estimated  according to their probabilty among other marks of each components and among experts
                                    and the most expected is chosen
            
            
        
    Returns:
        pd.DataFrame: dataframe with flags.
    """
    

    
    def all_ones(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        ann_ones_ones=ann_ones_group.apply(lambda x: x>0, axis=1)
        
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones_ones[col] = ann_ones_ones[col].astype(int)
            
        return  ann_ones_ones
            
    
    def intercept_ones(ann):
        columns=ann.columns
        columns_of_states=columns[3:]
        
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        #Здесь пока захардкожено 2
        ann_ones_intercept=ann_ones_group.apply(lambda x: x==2, axis=1)
        
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones_intercept[col] = ann_ones_intercept[col].astype(int)
            
        return  ann_ones_intercept
    
    
    
    def weigths_of_ones(ann):
    
        
        columns=ann.columns
        columns_of_states=columns[3:]
        
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        #Здесь пока захардкожено 2
        ann_weights_ones=ann_ones_group.apply(lambda x: x/sum(x), axis=1)
        
        return  ann_weights_ones    
    
    
        
    def weights(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]    
        
        ann_probs=ann.apply(lambda x:  (x[3:]/sum(x[3:]) if sum(x[3:])!=0 else x[3:] ), axis=1)
        ann_probs['ic_id']=ann['ic_id']
        #ann_probs['user_hash']=ann['user_hash']
        
        ann_probs_group=ann_probs.groupby(['ic_id']).sum()
        
        ann_probs_probs=ann_probs_group.apply(lambda x: x/sum(x), axis=1)
        
        return ann_probs_probs
    
    
    
    def weigths_with_desicion(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]    
        
        ann_probs=ann.apply(lambda x:  (x[3:]/sum(x[3:]) if sum(x[3:])!=0 else x[3:] ), axis=1)
        ann_probs['ic_id']=ann['ic_id']
        #ann_probs['user_hash']=ann['user_hash']
        
        ann_probs_group=ann_probs.groupby(['ic_id']).sum()
        
        ann_probs_probs=ann_probs_group.apply(lambda x: x/sum(x), axis=1)
        ann_probs_probs_with_desicion=ann_probs_probs.apply(lambda x: x==max(x), axis=1)
        
        return ann_probs_probs_with_desicion   
    
    

    if agg_type=='all_ones':
        
        df=all_ones(annotations)
        
    elif agg_type=='intercept_ones':
        
        df=intercept_ones(annotations)
        
    elif agg_type=='weigths_of_ones':
        
        df=weigths_of_ones(annotations)
    
    elif agg_type=='weights':
        
        df=weights(annotations)
        
    elif agg_type=='weigths_with_desicion':
        
        df=weigths_with_desicion(annotations)
    
    
    
    return df[flag]
    
    