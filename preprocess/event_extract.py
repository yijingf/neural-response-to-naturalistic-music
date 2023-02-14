import os
import mne
import pickle
import numpy as np

from utils import log_event_parser

import sys
sys.path.append('../config/')
from config import data_dir, preprocessed_data_dir, sfreq, filterbanks

freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'ripple']

def event_filter(prefix, start, end, freq_bands=[]):
    """
    ECoG data segmentation based on stimuli and save to .npz file.
    Parameters
    ----------
    prefix: str
        File name of the .fif file.
    start: int
        the start timestamp of the event in (time*sfreq)
    end: int
        the end timestamp of the event in (time*sfreq)

    Returns
    -------
    bp_data: dictionary
        {frequency_bands: data}

    Notes
    -----


    """
    # Load Data
    fname = os.path.join(preprocessed_data_dir, f'{prefix}.raw.fif')
    if not os.path.exists(fname):
        raise(f"{fname} doesn't exist.")
    raw = mne.io.read_raw_fif(fname, verbose=False)

    bp_data = {}
    data, _ = raw[:, start:end]
    bp_data['orig'] = data

    for bp_filter in freq_bands:
        lfreq, hfreq = filterbanks[bp_filter]
        bp_data[bp_filter] = mne.filter.filter_data(data, sfreq, lfreq, hfreq, verbose=False)

    return bp_data

def main(sub_sess, stimulus='K448orig_120', silence=False, all_freq=False):
    prefix = f'sub{sub_sess}'

    # Load stimulus start, end time
    event_fname = os.path.join(preprocessed_data_dir, f'{prefix}_event.npz')
    if not os.path.exists(event_fname):
        raise ValueError(f"{event_fname} doesn't exist.")
    event = np.load(event_fname)

    index = np.where(event['stimuli'] == stimulus)
    if not index[0].size:
        raise(f"{stimulus} not found")
    start = event['start'][index][0]
    end = event['end'][index][0]

    if silence:
        # Get ECoG corresponding to washed out period before/after the stimulus
        before_start = start - sfreq * 30
        before_end = start
        bp_data = event_filter(prefix, before_start, before_end)
        output_dir = os.path.join(preprocessed_data_dir, f'{stimulus}_before')
        os.makedirs(output_dir, exist_ok=True)
        output_fname = os.path.join(output_dir, f'{prefix}.npz')
        np.savez(output_fname, **bp_data)

        start += sfreq * 2 * 60
        end = start + sfreq * 1 * 30
        bp_data = event_filter(prefix, start, end)
        output_dir = os.path.join(preprocessed_data_dir, f'{stimulus}_after')
        os.makedirs(output_dir, exist_ok=True)
        output_fname = os.path.join(output_dir, f'{prefix}.npz')
        np.savez(output_fname, **bp_data)
        return

    if all_freq:
        bp_data = event_filter(prefix, start, end, freq_bands)
    else:
        bp_data = event_filter(prefix, start, end)

    output_dir = os.path.join(preprocessed_data_dir, stimulus)
    os.makedirs(output_dir, exist_ok=True)
    output_fname = os.path.join(output_dir, f'{prefix}.npz')
    np.savez(output_fname, **bp_data)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_sess', type=str, help='Subject ID and Session ID')
    parser.add_argument('--all_freq', dest='all_freq', default=False, action='store_true', help='Apply bandpass filters for all 7 frequency bands.')
    parser.add_argument('--stimulus', dest='stimulus', type=str, default='K448orig_120', help='K448orig_120, violetNoise, Wagner')
    parser.add_argument('--silence', dest='silence', default=False, action='store_true', help='washout period before and after each trial')

    args = parser.parse_args()
    main(args.sub_sess.upper(), args.stimulus, args.silence, args.all_freq)
