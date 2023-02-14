# iEEG Preprocessing Pipeline


## General Preprocessing
1. Apply bandpass filters 1-250 Hz and Notch filter at 60 Hz
2. Downsample the signal to 256 Hz
3. Reject bad channels with Z-score > 2.5

Usage:
```
python3 preprocess.py [-c CH_DIR] [-l LOG_DIR] [session ID]
```

Example
```
python3 preprocess.py 1A -c ../Data/SS2_LOCS/subject1.xlsx -l ../Data/SS2_LOGS/sub1A.log
```

## Data Segmentation
Extract iEEG epochs corresponding to the stimuli, e.g. `K448orig_120`, `violetNoise`, `Wagner` and etc.

Usage: 
```
event_extract.py [--stimulus STIMULUS] [--silence] [session ID] [baseline period]
```

Example
```
python3 event_extract.py 1A --stimulus K448orig_120 --silence before
```

## Power Spectral Density
Extract Power Spectral Density (PSD) with frequency resolution of 1Hz for each window. 

Usage:
```
python3 PSD_extract.py [--stimulus STIMULUS] [--t_win T_WIN] [session ID]
```
Example
```
python3 PSD_extract.py 1A --stimulus K448orig_120
```