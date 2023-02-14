# Evoked Response to Phrase Bondaries

## Onset Extraction
* Note Onset: Extracted using librosa.
* Phrase Boundary: Manually labeled by music experts.

Ussage
```
python3 stimuli_marker.py
```

## Evoked-Related Potential (ERP)
Average the iEEG epochs sampled around the stimuli (-200 ms to 600 ms) after rejecting epochs containing artifacts. These epochs were grouped by brain regions.

Usage:
```
python3 ERP_extract.py [Session ID] [-d output path] [-m type of onset] [--s stimulus] 
```

## Statistical Analysis
We verify the existence of ERP elicited by different musical components, and compare the distribution of these ERPs. An ERP is essentially a time series signal. Therefore, a [cluster-based permutation test](https://www.sciencedirect.com/science/article/pii/S0165027007001707?via%3Dihub) is used to compare two time series.

Usage:
```
# Run statistical analysis and save the results to file
python3 ERP_compare.py [-s Session ID] [-m mode]

# `mode` is either verifying ERPs or comparing two types of ERP
```

