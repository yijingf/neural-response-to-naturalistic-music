
# Neural Response to Naturalistic Music

A pilot study that explores different types of neural response to naturalistic music, and identify potential acoustic features and biomarkers associated with Mozart's Effect. 

## Baseline Neural Response Classifier
Classifiy the neural response to different stimuli. See more details in `./classification`.

* Classifier: LDA
* Feature Extraction: Average bandpower of 7 frequency bands (delta, theta, alpha, beta, gamma, high gamma, ripple).
* Baseline Performance (Binary Classification): 74.03\%.


## Evoked Response to Phrase Boundaries
* Compare the evoked responses to phrase boundaries and note onsets, see `./ERP` and [here](https://www.nature.com/articles/s41598-022-13710-3) for more details.
* Acoustic Feature: Note onsets and manually labled phrase boundaries.
* Neural Response: Event-related potential (ERP).
* Statistical Analysis: A cluster-based permutation test to compare the ERP curves.


## Data
Deidentified Stereo-EEG. See `./preprocess` for more details.



