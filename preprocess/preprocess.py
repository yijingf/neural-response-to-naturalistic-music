import os
import re
import mne
import numpy as np
from glob import glob

from utils import *

# Config
import sys
sys.path.append('../config/')
from config import raw_dir, data_dir, preprocessed_data_dir

def preprocess(raw, ds_freq, reject_thresh=2.5):
    """Apply band-pass filters, down sampling and reject bad channels with Z-score > 2.5.
    Parameters
    ----------
    raw: mne data
        ECoG data
    sfreq: int
        the original sampling frequency of ECoG data
    ds_freq: int
        downsampling frequency

    Returns
    -------
    picks: mne data
        preprocessed ECoG data in mne data format
    start: array
        an array of starting time of each trial
    end: array
        an array of ending time of each trial

    Notes
    -----

    """
    # Bandpass filter
    raw = filters(raw)

    # Downsample
    if raw.info['sfreq'] != ds_freq:
        raw = downsample(raw, ds_freq)

    # Get events start/end timestamp (time*ds_freq)
    start, end = get_event_time(raw)

    # Reject bad channels
    bad_chs = check_bads_adaptive(raw)
    raw.info['bads'] = bad_chs
    # check which channels are marked as bad
    print(raw.info['bads'])
    print(f'BAD CHANNELS = {len(bad_chs)}', end="\r", flush=True)

    picks = raw.pick_types(eeg=True, meg=False, exclude='bads')
    print('Finished rejecting bad chans (> 2.5 SD): picks = clean data.')
    print("CHECK: ZERO = {}\n".format(len(picks.info['bads'])))
    return picks, start, end

def main(sub_sess, ch_dir, log_dir, down_sfreq=256):
    """ECoG data preprocessing pipeline.
    Preprocess ECoG data, extract target events from log and save the data to .fif files. 

    Parameters
    ----------
    sub_sess: str
        Subject ID and Session ID
    ch_dir: str
        directory to the excel file that stores electrodes information
    log_dir: str
        directory to the log file
    down_sfreq: int
        downsampling frequency, default is 256 Hz.

    Note
    -------
    No returns. Save the preprocessed ECoG data to 'sub[sub_sess].fif' and the timestamps corresponding to music excerpts to 'sub[sub_id]_[trial]_[down_sfreq].npz' file under the 'preprocessed_data_dir' directory.

    """
    edf_fname = os.path.join(raw_dir, f'sub{sub_sess}.EDF')


    # Load .edf file
    raw = mne.io.read_raw_edf(edf_fname, preload=True)
    channels = raw.ch_names

    # Find stimulation channel
    channels = raw.ch_names
    if 'DC1' in raw.ch_names:
        stim_channel = 'DC1'
    elif 'DC01' in raw.ch_names:
        stim_channel = 'DC01'
    else:
        raise ValueError(f"{edf_fname} has no stimulation channel.")

    # Load channel location info from Excel file
    sub_id = re.findall(r'\d+', sub_sess)
    fname = os.path.join(ch_dir, f'subject{sub_id}.xlsx')
    xl = pd.ExcelFile(fname)
    sheet_names = xl.sheet_names  # see all sheet names
    locs = xl.parse(sheet_names[-1], header=None)  # read a specific sheet to DataFrame
    
    excel_ch = [i.replace("'", '') for i in locs[0]]
    excel_ch = [i[:-2] + str(int(i[-2:])) for i in excel_ch]

    edf_ch = pd.Series(channels)
    selected_ch_index = edf_ch[edf_ch.isin(excel_ch)].index.to_numpy()
    selected_chs = np.array(channels)[selected_ch_index]

    # Remove irrelevant channels from the .edf file
    remove_ch = list(set(channels).difference(set(selected_chs)))

    # Keep stimulation channel
    if stim_channel in remove_ch:
        remove_ch.remove(stim_channel)
    picks = raw.pick_types(eeg=True, meg=False, exclude=remove_ch)

    # Bandpass filter, reject bad channels
    picks, start, end = preprocess(picks, down_sfreq)

    # Load events info from log
    log_fname = os.path.join(log_dir, f'sub{sub_sess}.log')
    if not os.path.exists(log_fname):
        raise ValueError(f"{log_fname} doesn't exist.")
    stimuli = log_event_parser(log_fname)

    # Save output
    output_fname = os.path.join(preprocessed_data_dir, f'sub{sub_sess}.raw.fif')
    raw.save(output_fname, overwrite=True)
    output_event = os.path.join(preprocessed_data_dir, f'sub{sub_sess}_event.npz')
    np.savez(output_event, start=start, end=end, stimuli=stimuli)
    
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_sess', type=str, default='1A', help='subject ID and Session ID')
    parser.add_argument('-c', '--ch_dir', dest='ch_dir', type=str, default='', help='Directory to channel location excel files')
    parser.add_argument('-l', '--log_dir', dest='log_dir', type=str, default='', help='Directory to the log file')

    args = parser.parse_args()

    ch_dir = args.ch_dir or os.path.join(data_dir, 'SS2_LOCS')
    log_dir = args.log_dir or os.path.join(data_dir, 'SS2_LOGS') 
    print(f'Reading channel information from {ch_dir}')

    main(args.sub_sess, ch_dir, log_dir, down_sfreq=256)