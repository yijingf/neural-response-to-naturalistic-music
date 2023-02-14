import os
import json
import numpy as np
from scipy import signal

from utils import get_trials, get_channel_info, sample_between_onsets

import sys
sys.path.append('../config/')
from config import sfreq, meta_dir, preprocessed_data_dir

# Stimuli Config
thresh = 1.2
t_window = 0.8
duration = int(t_window * sfreq)

# Load Stimulus K448
stimuli = np.load('stimuli.npz')
struct_time = stimuli['struct']
onset_time = stimuli['onset']

def reject_spike_events(spikes, events, reverse=False):
    """Reject or keep events overlap with at least one spike in one channel.

    Parameters
    ----------
    spikes: dict, {ch: location}
    events: 1D numpy array
        
    reverse: bool
        Keep all the trials that contain at least one spike if reverse is True. 

    Returns
    -------
    selected_events: 1D numpy array
        beat time in second

    Notes
    -----

    """
    if not spikes.any():
        if reverse:
            return np.arrray([])
        return events
    cond = []
    for spike in spikes:
        cond0 = events <= spike[0]
        cond1 = events >= spike[1]
        cond.append(np.bitwise_xor(cond0, cond1))
        
    cond = np.sum(np.stack(cond), axis=0)
    if reverse:
        selected = np.where(cond != len(spikes)) 
    else:
        selected = np.where(cond == len(spikes)) 

    selected_events = events[selected]
    return selected_events

def spike_integrate(spikes, cnt_shape):
    """Integrate spikes from all channels.
    Parameters
    ----------
    spikes: dict, {ch: location}
    cnt_shape: tuple

    Returns
    -------
    integrated_spikes: 1D numpy array
        beat time in second

    Notes
    -----

    """
    cnt = np.zeros(cnt_shape)
    for _, ch_spikes in spikes.items():
        for spike in ch_spikes:
            cnt[spike[0]:spike[1]] = 1
    
    cnt_diff = np.diff(cnt)
    st = np.where(cnt_diff == 1)[0]
    ed = np.where(cnt_diff == -1)[0]
    
    integrated_spikes = np.array([st, ed]).T
    return integrated_spikes


def get_erp_window(prefix, events=[], stimulus='K448orig_120', normalize=False, spikes_dir="spikes_orig"):
    # Load Data
    fname = os.path.join(preprocessed_data_dir, stimulus, f'{prefix}.npz')
    data = np.load(fname)['orig']

    data = signal.detrend(data)
    _, n_sample = data.shape

    # Data Normalization (Deprecated)
    if normalize:
        data = data/np.expand_dims(np.linalg.norm(data, axis=1), axis=1)

    if not len(events):
        n_trials = int((n_sample - 0.2 * sfreq)/duration)-1 # 0.2 s before the event
        events = np.arange(n_trials) * t_window + 0.2
        
    # Spike rejection
    with open(f'{spikes_dir}/{stimulus}/{prefix}.json', 'r') as f:
        spikes = json.load(f)
    spikes = spike_integrate(spikes, n_sample)
    events = reject_spike_events(spikes, events)
    
    # ch * n_trial * time points before averaging
    windows = get_trials(data, events)

    return windows

def main(sub_sess, output_dir, stimulus='K448orig_120', mode="onset", silence=False):
    assert mode in ["onset", "struct", "reference"]

    prefix = f'sub{sub_sess}'

    if silence:
        # extract washout period after the stimulus
        assert (mode == 'reference' and silence)
        windows = []
        for stimulus in ["K448orig_120_before", "K448orig_120_after"]:
            window = get_erp_window(prefix, stimulus=stimulus)
            windows.append(window)
        # todo?
        windows = np.concatenate(windows)
        output_dir = os.path.join(output_dir, "silence")

    if stimulus == 'violetNoise':
        events = []
        output_dir = os.path.join(output_dir, "noise")

    if mode == "reference":
        events = sample_between_onsets(onset_time)
        output_dir = os.path.join(output_dir, mode)
    else:
        events = stimuli[mode]
        output_dir = os.path.join(output_dir, mode)

    os.makedirs(output_dir, exist_ok=True)

    windows = get_erp_window(prefix, events=events, stimulus=stimulus)

    # Loading channel location info
    loc_fname = os.path.join(meta_dir, 'LOCS', f'{prefix}.json')
    cortex, loc = get_channel_info(loc_fname)

    # Group erp/blank data by lobe
    window_lobe = {}

    for i, lobe in enumerate(cortex):
        chs = loc[lobe]
        window_lobe[lobe] = windows[chs]
    
    output_fname = os.path.join(output_dir, prefix)
    print(f'Saving to file: {output_fname}.\n')
    np.savez(output_fname, **window_lobe)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # def main(sub_sess, output_dir, stimulus='K448orig_120', mode="onset", silence=False):
    parser.add_argument('sub_sess', type=str, help='Subject ID and Session ID')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, help='output directory')
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='onset', help='noise, onset, struct, reference')
    parser.add_argument('--stimulus', dest='stimulus', type=str, default='K448orig_120', help='violetNoise, K448orig_120, Wagner')
    parser.add_argument('--silence', dest='silence', default=False, action='store_true', help='washout period before and after each trial')

    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join('./res/plot_data')
    if not args.sub_sess:
        raise "No valid subjectID and Session ID."
    main(args.sub_sess.upper(), output_dir, args.stimulus, args.mode, args.silence)