import os
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from scipy import signal

from cluster_test import permutation_cluster_test

import matplotlib as mpl
mpl.use('Agg')

# Config
import sys
sys.path.append('../config/')
from config import root_dir, sfreq

# Other Config
res_dir = './res'
os.makedirs(res_dir, exist_ok=True)

thresh = 1.2
duration = int(0.8*sfreq)

def plot_erp(trial_lobe, ref_lobe, cortex, fig_fname, mode='onset'):
    time_axis = np.arange(duration)/sfreq*1000-200
    
    n_plot = len(cortex)
    n_cols = 3
    n_rows = int(np.ceil(n_plot/n_cols))

    fig, ax = plt.subplots(nrows=n_rows, sharex=True, sharey=True, ncols=n_cols, figsize=(12, int(2.5*n_rows)))
    # fig.tight_layout()
    
    if 'vs_onset' in mode:
        event_label = 'Structural Boundary'
        ref_label = 'Note Onset'
    else:
        if mode == 'onset':
            event_label = 'Note Onset'
        elif mode == 'struct':
            event_label = 'Structural Boundary'
        else:
            event_label = 'Events'
        ref_label = 'Reference'
        
    for i, lobe in enumerate(cortex):
        r, c = i//n_cols, i%n_cols
        lower_bound = np.max(trial_lobe[i])*1.5
        upper_bound = np.max(trial_lobe[i])*1.5
        ax[r, c].grid()
        ax[r, c].plot(time_axis, trial_lobe[i], label=event_label)
        ax[r, c].plot(time_axis, ref_lobe[i], label=ref_label)
        ax[r, c].set(title=lobe, ylabel=None)
        # ax[r, c].vlines([0], lower_bound, upper_bound, alpha=0.5, color='r', linestyle='--')

    handles, labels = ax[r, c].get_legend_handles_labels()

    # Add legend
    fig.add_subplot(111, frameon=False)
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    # hide tick and tick label of the big axis

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time/ms", fontsize=15)
    plt.ylabel('Average Amplitude', fontsize=15)

    plt.title("Average Normalized ECoG Response for Each Lobe", y=1.05, fontsize=15)
    # plt.show()
    plt.savefig(fig_fname, bbox_inches = 'tight')
    plt.close()

def plot_perm_test(clusters, cluster_ps, cortex, T_obs_s, fig_fname):
    time_axis = np.arange(duration)/sfreq*1000-200
    n_plot = len(cortex)
    n_cols = min(3, len(cortex))
    n_rows = int(np.ceil(n_plot/n_cols))
    fig, ax = plt.subplots(nrows=n_rows, sharex=True, ncols=n_cols, figsize=(12, int(2.5 * n_rows)))

    for i, lobe in enumerate(cortex):
        r, c = i//n_cols, i%n_cols
        for i_c, c0 in enumerate(clusters[i]):
            c0 = c0[0]
            if cluster_ps[i][i_c] <= 0.05:
                h = ax[r, c].axvspan(time_axis[c0.start], time_axis[c0.stop - 1], color='r', alpha=0.3, linewidth=0)
        ax[r, c].grid()
        ax[r, c].plot(time_axis, T_obs_s[i], 'b')
        # ax[r, c].legend(('cluster p-value < 0.05',))
        ax[r, c].set(title=lobe, ylabel=None)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time/ms", fontsize=15)

    # Z-score by default 
    # plt.ylabel('T-statistics', fontsize=15)
    plt.ylabel('Z-score', fontsize=15)

    plt.title("Permutation Test", y=1.1, fontsize=15)    
    # plt.show()
    plt.savefig(fig_fname)
    plt.close()

def get_ChannelInfo(loc_fname):
    """
    Return:
        cortex
        loc
    """
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
