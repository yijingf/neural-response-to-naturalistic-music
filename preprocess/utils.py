"""
ECoG signal preprocessing is based on Robert Quon's code.
"""

import os
import mne
import numpy as np
import pandas as pd
from scipy import stats

def log_event_parser(fname):
    """
    Parse music event labels and timestamps from the log file.
    Return a pandas data frame contains music, start time, and end time.

    Parameters
    ----------
    fname: string
        file name of the .log file. 

    Returns
    -------
    events: pandas dataframe
        start time, end time and the name of music excerpts

    Notes
    -----
    """
    with open(fname, 'r') as f:
        orig_data = f.read().splitlines()

    # Get info of each trial
    music_event = [i.split(' -- ') for i in orig_data if 'SoundEvent' in i and 'Resp' not in i]
    music_event = list(map(list, zip(*music_event)))
    music = [os.path.basename(i).replace(".wav", "") for i in music_event[1]]
    return music

    """
    # Deprecated
    # Get start/end time of each trial (using the start time of SART as the end of music event)
    # Probably don't need this in the future
    start_time = [float(i.split(': ')[-1]) for i in music_event[-1]]
    sart = [i for i in orig_data if 'SART' in i]
    end_time = [float(i.split(": ")[-1]) for i in sart]
    
    from datetime import datetime, timedelta
    # Time zone
    time_diff=0
    if time_diff:
        start_time = np.array([datetime.fromtimestamp(i) for i in start_time]) - timedelta(hours=time_diff)
        end_time = np.array([datetime.fromtimestamp(i) for i in end_time]) - timedelta(hours=time_diff)
    
    events = pd.DataFrame({"Event": music, "Start Time":start_time, "End Time": end_time})
    return events
    """

def get_event_time(raw, duration=90, duration_max=100):
    """
    Get start/end timestamp of the event in time*sfreq
    Parameters
    ----------
    raw: mne data
        ECoG data
    duration: (optional) int
        duration of target event in seconds, default is 90
    duration_max: (optional) int
        the maximum duration of target event in seconds, default is 100

    Returns
    -------
    start: a list of int
        the list of events start time in seconds
    end: a list of int
        the list of events end time in seconds
    """
    sfreq = raw.info['sfreq']

    # Find stimulation channel
    channels = raw.ch_names
    if 'DC1' in raw.ch_names:
        stim_channel = 'DC1'
    elif 'DC01' in raw.ch_names:
        stim_channel = 'DC01'
    else:
        raise ValueError("Please set up stimulation channel manully.")

    events = mne.find_events(raw, stim_channel=stim_channel, min_duration=1/sfreq, consecutive=False, output='offset')

    timestamp = events[:,0]
    diff = timestamp[1:] - timestamp[:-1]

    index = np.where((diff >= duration * sfreq) & (diff <= duration_max * sfreq) )
    start = timestamp[index]
    end = timestamp[index[0]+1]
    return start, end

def filters(raw):
    """Notch filter at 60 Hz, Bandpass filter 1-250 Hz

    Parameters
    ----------
    raw: mne data
        ECoG data

    Returns
    -------
    raw_filtered: mne data
        filtered ECoG data

    Notes
    -----

    """
    # notch filter
    raw_n = raw.notch_filter(np.arange(60, 241, 60), filter_length='auto', phase='zero')

    # low pass filter (250Hz)
    raw_nl = raw_n.filter(None, 250., h_trans_bandwidth='auto', filter_length='auto',
               phase='zero')

    # high pass filter (1Hz) - remove slow drifts
    raw_filtered = raw_nl.filter(1.0, None, l_trans_bandwidth='auto', filter_length='auto', # 1.0
               phase='zero')
    print('Finished notch filter, low pass filtering, high pass filtering!\n')
    return raw_filtered

def downsample(raw, ds_freq=500):
    """
    Signal downsampling.
    raw: mne
    """
    ### downsampling (500Hz) --- check that all initial SF >500Hz
    # downsample to 500Hz
    raw_ds = raw.resample(ds_freq, npad='auto')
    print('Finished downsampling.')
    print('New sampling rate:', raw_ds.info['sfreq'], 'Hz\n')
    return raw_ds

def check_bads_adaptive(raw, thresh=2.5, max_iter=np.inf): 
    """Reject bad electrodes based on variance. 

    Parameters
    ----------
    raw: mne data

    thresh: float
        Z-score threshold for rejecting bad channels, default is 2.5

    max_iter: int
        maximum iteration, default is infinity
    
    Returns
    -------
    bad_chs: list
        a list of bad channels

    """
    # ch_x = fun(raw[:,0], axis=-1)
    ch_x = np.var(raw[:,0], axis=-1)
    my_mask = np.zeros(len(ch_x), dtype=np.bool)
    i_iter = 0
    while i_iter < max_iter:
        ch_x = np.ma.masked_array(ch_x, my_mask)
        this_z = stats.zscore(ch_x)
        local_bad = np.abs(this_z) > thresh
        my_mask += local_bad.flatten()
        print('iteration %i : total bads: %i' % (i_iter, sum(my_mask)), end="\r", flush=True)
        if not np.any(local_bad):
            break
        i_iter += 1
    bad_chs = np.array(raw.ch_names)[my_mask]
    return list(bad_chs)