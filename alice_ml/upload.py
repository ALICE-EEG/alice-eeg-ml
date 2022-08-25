import mne
import numpy  as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")


def raw_to_epoch(raw):
    events=[]
    i=0
    while i*3*raw.info['sfreq']<len(raw.times):

        event=[i*3*raw.info['sfreq'],0,1]
        events.append(event)
        i=i+1

    events_ar=np.array(events).astype(int)[:-1]
    return mne.Epochs(raw, events_ar, event_id=[1], tmin=0, tmax=3.0, baseline=(0,0))


def generate_token(username,password):
    url = 'http://alice.adase.org/api/auth'
    # url = 'http://localhost:8000/api/auth'

    data = {
      'username': username,
      'password': password
    }

    r = requests.post(url, data)

    token = r.json()['token']
    headers = {'Authorization': "Token {}".format(token)}
    return headers, data


def upload_objects(ica, raw_epoch, d, dataset, username, password):
    '''
    Uploads ICA components to Alica website 
    
    Args:
        ica - ICA components
        raw_epoch - data, can be raw format or epoch format
        d - name of subject
        dataset - name of dataset
        username, password - data to log in to Alice
    
    '''
    headers, data_ = generate_token(username,password)    
    
    if isinstance(raw_epoch, mne.io.BaseRaw):
        epochs_short = raw_to_epoch(raw_epoch)
    else:
        epochs_short=raw_epoch

    channels_to_use = [ch.lower() for ch in ica.info['ch_names']]
    ica_src_df = ica.get_sources(epochs_short).to_data_frame().reset_index()

    url = 'http://alice.adase.org/api/data/ic'

    r = requests.post(url, data_)
    data_base = {
      "subject": d,
      "dataset": dataset,
      "sfreq": epochs_short.info['sfreq'],
    }

    n_components=ica.n_components
    for ic_idx in range(ica.n_components):

        try:
            df_weights = pd.DataFrame({'ch_name': channels_to_use, 'value': ica.get_components()[:, ic_idx]})
            col_name = f'ICA{ic_idx:03.0f}'
            df_data = ica_src_df[['epoch', col_name]].rename(columns={col_name: 'value'})
            data = data_base.copy()
            data.update({
              'name': f'IC{ic_idx:03.0f}',
              'data': {
                  'ica_weights': df_weights.to_dict(orient='list'),
                  'ica_data': df_data.to_dict(orient='list')
              }
            })
            r = requests.post(url, json=data, headers=headers)
            assert (r.status_code == 201)
            print(ic_idx)

        except:
            print('Не загрузилось компонент '+str(ic_idx))