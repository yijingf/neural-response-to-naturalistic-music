"""
Configuration
"""
import os

# Sampling frequency
sfreq = 256

# Directories
curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(curr_dir)

raw_dir = root_dir # directory to raw iEEG data
meta_dir = root_dir # directory to meta data, i.e. electrodes location
preprocessed_data_dir = os.path.join(root_dir, 'Preprocessed_Data') # directory to processed data


# Frequency Ranges
filterbanks = {'delta':[2,4], 
               'theta':[4,7],
               'alpha':[8,12], 
               'beta':[12, 30], 
               'gamma':[30, 40],
               'high_gamma':[40, 80], 
               'ripple':[80, 120], 
               'lower_beta':[2, 12],
               'higher_beta':[12, 120], 
               'lower_gamma':[2, 30], 
               'higher_gamma':[30, 120], 
               'lower_high_gamma':[2, 40], 
               'higher_high_gamma':[40, 120], 
               'lower_60':[2,60],
               'high_60':[60, 120]}