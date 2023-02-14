import os
import json
import numpy as np

def get_channel_info(loc_fname):
    """Load channel, lobe information.
    Parameters
    ----------
    loc_fname: str 
        path to channel location .json file
    
    Returns
    -------
    cortex: list
        list of cortices
    loc: dict 
        {cortex: [ch_index]} index of electrodes in each lobe
    
    Notes
    -----
    """
    
    # Input sanity check
    ext = os.path.basename(loc_fname).split('.')[-1]
    assert ext == 'json', ".json file is required"

    with open(loc_fname, 'r') as f:
        loc = json.load(f)
    
    cortex = [i for i in loc.keys() if 'white matter' not in i.lower() and 'out of' not in i.lower()]
    
    intersect = [item for i, item in enumerate(cortex) if '/' in item]
    for i in intersect:
        items = i.split('/')
        for item in items:
            item = item.strip().capitalize()
            loc[item] = loc.get(item, []) + loc[i]
        cortex.remove(i)
        
    temporal_cortex = [i for i in cortex if 'temporal' in i.lower()]
    frontal_cortex = [i for i in cortex if 'frontal' in i.lower()]
    
    cortex += ['Temporal Lobe', 'Frontal Lobe']
    loc['Temporal Lobe'] = list(np.concatenate([loc[i] for i in temporal_cortex]))
    loc['Frontal Lobe'] = list(np.concatenate([loc[i] for i in frontal_cortex]))
    
    for i in cortex:
        loc[i] = list(set(loc[i]))

    return cortex, loc

def sampling(data, n_trials=200):
    """Data downsample/upsample.

    Parameters
    ----------
    data: 2D numpy array 
        # trials * time-series
    n_trials: int
        output # of trials 
    
    Returns
    -------
    data: 2D numpy array
        resampled data
    
    Notes
    -----

    """
    n = len(data)
    if n > n_trials:
        # downsampling
        idx = np.random.choice(n, n_trials, replace=False)
        data = data[idx]
    elif n < n_trials:
        # upsampling
        data = np.repeat(data, int(n_trials/n), axis=0)
        if n_trials%n:
            idx = np.random.choice(n, n_trials%n, replace=False)
            data = np.vstack((data, data[idx]))
    return data

def sample_between_onsets(onset_time, t_window=0.8, sfreq=256):
    """Randomly sample events between two note onsets
    Parameters
    ----------
    onset_time: 1D numpy array
        timestamp of note onsets in second
    t_window: float
        duration of ECoG window in second
    sfreq: int
        sampling frequency 

    Returns
    -------
    events: 1D numpy array
        timestamp of randomly sampled events in second

    Notes
    -----
    """
    interval = np.diff(onset_time)
    interval_index = np.where(interval >= t_window)[0]
    
    n_per_interval = np.round(interval[interval_index]/t_window).astype(int)
    range_per_interval = interval[interval_index] - t_window
    
    events = []
    for i, v in enumerate(interval_index):
        events.append(np.random.choice(np.arange(onset_time[v], onset_time[v]+range_per_interval[i], 1/sfreq), 
                         n_per_interval[i], replace=False))
    events = np.concatenate(events)
    return events

def get_trials(data, events, t_window=0.8, st=-0.2, sfreq=256):
    """
    Extract ECoG window [-200ms, 800ms] given the entire time-series and events. 
    Parameters
    ----------
    data: 2D numpy array (ch * time points)
        ECoG data
    events: 1-D array
        Series of event time in second.
    t_window: float
        duration of ECoG window in second
    st: float
        window start time. default is -200 ms before the event.
    sfreq: int
        sampling frequency
    
    Returns
    -------
    trials: 3D numpy array
        ch * n_trial * time points
    
    Notes
    -----
    """
    duration = int(sfreq * t_window)
    n_ch, _ = data.shape
    if not events.size:
        return np.empty((n_ch, ))
    
    trials = []
    for i, ch in enumerate(data):
        epoch = np.stack([ch[int((i+st) * sfreq):int((i+st) * sfreq)+duration] for i in events])
        trials.append(epoch)
    trials = np.array(trials)
    return trials