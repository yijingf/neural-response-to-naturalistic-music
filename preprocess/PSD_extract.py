import os
import mne
import pickle
import numpy as np

from scipy import signal

import sys
sys.path.append('../config/')
from config import preprocessed_data_dir, sfreq


def getSlidingWindow(data, t_win, sfreq, overlap=0.5):
    """
    Trunk data into sliding window with overlapping
    
    Args:
        data: np.array
        t_win: length of window in second
    """    
    _, len_signal = data.shape
    s_win = int(t_win*sfreq)
    step = int(s_win*overlap)

    n_win = int(np.floor((len_signal - s_win)/step) + 1)
    wins = [[i*step, i*step+s_win] for i in range(n_win)]
    signal_win = [data[:,st:ed] for st, ed in wins]
    return signal_win

def getPSD(data, min_freq=2):
    """
    Get PSD
    (Adapted from Robert's code.)
    """
    # freqs, psd = signal.welch(data, sf, nperseg=int(0.9*sfreq), detrend='linear', scaling='spectrum', average='median')

    win_len = 2/min_freq
    # take a window sufficiently long to encompasses at least 2 full cycles of the lowest frequency of interest
    s_win = sfreq*win_len
    freqs, psd = signal.welch(data, sfreq, nperseg=s_win, detrend='linear') # default overlap is s_win/2
    return freqs, psd


def main(sub_sess, stimulus, t_win=10, overwrite=False):
    fname = os.path.join(preprocessed_data_dir, stimulus, f'sub{sub_sess}.npz')
    data = np.load(fname)['orig']
    
    data_window = np.stack(getSlidingWindow(data, t_win, sfreq))
    psd_window = [getPSD(i) for i in data_window]
    freqs = psd_window[0][0]
    psd_window = np.stack([i[1] for i in psd_window])
    
    output_dir = os.path.join(preprocessed_data_dir, 'PSD')
    os.makedirs(output_dir, exist_ok=True)

    output_fname = os.path.join(output_dir, f'sub{sub_sess}.npz')

    if os.path.exists(output_fname):
        output_data = dict(np.load(output_fname))
        if not overwrite and stimulus in output_data:
            raise ValueError(f"PSD for {stimulus} already exists in {output_fname}")
    else:
        output_data = {}

    output_data[stimulus] = psd_window
    np.savez(output_fname, **output_data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_sess', type=str, help='Subject ID and Session ID')
    parser.add_argument('-o', dest='overwrite', default=False, action='store_true', help='Overwrite exisiting PSD.')
    parser.add_argument('--stimulus', dest='stimulus', type=str, default='K448orig_120', help='K448orig_120, violetNoise, Wagner')
    parser.add_argument('--t_win', dest='t_win', type=float, default=10, help='Duration of Sliding window in seconds. Default is 10.')

    args = parser.parse_args()
    main(args.sub_sess.upper(), args.stimulus, args.t_win, args.overwrite)