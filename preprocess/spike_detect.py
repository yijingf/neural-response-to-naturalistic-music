import os
import json
import numpy as np

# function provided by Robert Quon
from spikedetector.detect import *

# Config
import sys
sys.path.append('../config')
from config import preprocessed_data_dir, sfreq

def main(sub_sess, freq_band='orig', stimulus='K448orig_120'):
    data_dir = os.path.join(preprocessed_data_dir, stimulus)
    prefix = f'sub{sub_sess}'

    fname = os.path.join(data_dir, f'{prefix}.npz')
    output_dir = os.path.join(f'spikes_{freq_band}', stimulus)
    os.makedirs(output_dir, exist_ok=True)
    output_fname = os.path.join(output_dir, f'{prefix}.json')

    data = np.load(fname)[freq_band]
    res = {}
    for i, channel in enumerate(data):
        tmp_res = detect(channel, samp_freq=sfreq, return_eeg=False)
        if tmp_res:
            tmp_res = np.stack(tmp_res).tolist()
        res[i] = tmp_res


    output_fname = os.path.join(output_dir, f'{prefix}.json')
    with open(output_fname, 'w') as f:
        json.dump(res, f)
    return

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sub_sess", type=str, help="[Subject ID][Session ID]")
    parser.add_argument('-s', '--stimulus', dest='stimulus', type=str, default='K448orig_120', help='K448orig_120, K448orig_120_before, K448orig_120_after, violetNoise, Wagner')
    parser.add_argument('-b', '--freq_band', dest='freq_band', type=str, default='orig', help='orig, alpha, beta, theta. etc')
    
    args = parser.parse_args()
    main(args.sub_sess, args.freq_band, args.stimulus)