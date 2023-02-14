import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../config/')
from config import root_dir

def get_beat_onset(data, fs, hop_length=1024, onset_beat=False):
    """
    Extract beat and onset from audio signal

    Parameters
    ----------
    data: 2D numpy array
        audio signal
    fs: int
        sampling frequency
    hop_length: int
        
    onset_beat: bool
        track beat from onset envelope

    Returns
    -------
    beat_time: 1D numpy array
        beat time in second
    onset_time: 1D numpy array
        onset time in second
    onset_env: 1D numpy array
        onset strength at each time point
    times: 1D numpy array
        time in second corresponding to onset_env

    Notes
    -----

    """
    onset_env = librosa.onset.onset_strength(y=data, sr=fs, hop_length=hop_length, aggregate=np.median)
    
    tempo, beats = librosa.beat.beat_track(y=data, sr=fs, hop_length=hop_length)
    if onset_beat:
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=fs)

    times = librosa.times_like(onset_env, sr=fs, hop_length=hop_length)
    onset_time = times[np.where(onset_env)]
    beat_time = times[beats]
    return beat_time, onset_time, onset_env, times

def load_stimuli(thresh=1.2, display=True):
    """
    Extract beat and onset from audio signal

    Parameters
    ----------
    thresh: float
        onset strength threshold, default is 1.2, which keeps 50% of the note onsets
    display: bool
        display stimuli

    Returns
    -------
    onset_time: 1D numpy array
        timestamp of note onsets in second
    struct_time: 1D numpy array
        timestamp of struct boundaries in second

    Notes
    -----
    """

    # Load Stimulus K448
    data_fname = os.path.join(root_dir, 'Stimulus' , 'K448orig_120.npz')
    K448_fs = 44100
    K448_signal, _ = librosa.load(data_fname, sr=K448_fs, duration=100, mono=True)

    # Extract onset
    _, onset_time, onsets, times = get_beat_onset(K448_signal, K448_fs, hop_length=1024)
    onset_time = times[np.where(onsets >= thresh)][:-1]
    onset_time = onset_time[np.where(onset_time < 90)]

    # Load K448 structure data
    struct_time = pd.read_csv('k448_structuralLabels.txt', delimiter='\t', header=None)[0].to_numpy()
    analysis =  ['intro', 'theme', 'continuation', 'descend seq',
                 'transit', 'secondary theme', 'continuation', 'sequential build']

    struct_time = struct_time[np.where(struct_time < 90)]
    structure = [i for i in zip(*(struct_time, analysis))]

    if display:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(times, onsets[:len(times)], label='Note Onsets')
        plt.axhline(y=1.2, xmin=0, xmax=1, color='red', linestyle='-.', label='Onset Threshold')
        plt.title('Note Onset and Structural Boundary', fontdict={'fontsize': 12})
        plt.vlines(struct_time, 0, 35, alpha=0.5, color='black', linestyle='--', label='Structural Boundary')
        plt.legend()
        plt.xlim(0, 90)
        plt.ylim(0, 35)
        plt.ylabel('Onset Strength', fontsize=12)
        plt.xlabel("Time/s", fontdict={'fontsize': 12})

        for timestamp, text in structure:
            plt.text(timestamp + 0.25, 20, text[:20])

        plt.savefig('stimuli')
        fname = os.path.join(os.path.abspath('.'), 'stimuli.png')
        print(f'Saving to {fname}')
        plt.close()

    return onset_time, struct_time
    
def main(display=True):
    onset_time, struct_time = load_stimuli(display)
    stimuli = {'struct':struct_time, 'onset':onset_time}
    np.savez('stimuli', **stimuli)
    fname = os.path.join(os.path.abspath('.'), 'stimuli.npz')
    print(f'Saving to {fname}')
    return

if __name__ == '__main__':
    main()