"""
Binary classification with permutation test
"""

import os
import json
import pickle
import numpy as np

from tqdm import tqdm
from scipy.integrate import simps
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys
sys.path.append('../config/')
from config import filterbanks, preprocessed_data_dir

# load label encoder
with open('event_label.pkl', 'rb') as f:
    le = pickle.load(f)

def get_bandpower(psd, freq_band, freqs):
    # Find values within the selected freq band
    idx = np.logical_and(freqs >= freq_band[0] , freqs <= freq_band[1])

    freq_resolution = freqs[1]-freqs[0]
    # Compute the absolute power by approximating the area under the curve
    bp = simps(psd[:,idx], dx=freq_resolution)        
    return bp

def get_all_bandpower(psd, freqs, freq_bands, relative=True):
    freq_resolution = freqs[1]-freqs[0]
    bps = [get_bandpower(psd, filterbanks[freq_band], freqs) for freq_band in freq_bands]
    bps = np.array(bps)
    if relative:
        total_bp = simps(psd, dx=freq_resolution)
        bps /= total_bp
    return bps

def load_data(fname, classes, freq_bands, psd_res=1):
    """
    psd_res: Frequency Resolution of psd
    """
    data = np.load(fname)
    X = []
    y = []

    for label, stimulus in zip(*(classes, stimuli)):
        psd_win = data[stimulus]
        freqs = np.arange(0, psd_win.shape[-1] * psd_res, psd_res)

        # Feature Extraction
        bp_win = np.array([get_all_bandpower(psd, freqs, freq_bands) for psd in psd_win])
        n,_,_ = bp_win.shape
        X.append(np.mean(bp_win, axis=2))
        y.append([label for _ in range(n)])

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y

def Kfold_acc_lda(X, y, n_fold=10):
    kf = KFold(n_splits=n_fold, shuffle=True)
    kf.get_n_splits(X)

    # print(kf)
    clf = LinearDiscriminantAnalysis()
    Y_pred = []
    Y_test = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        Y_pred.append(y_pred)
        Y_test.append(y_test)

    Y_pred = np.concatenate(Y_pred)
    Y_test = np.concatenate(Y_test)
    acc = np.sum(Y_pred==Y_test)/len(Y_test)
    #     print('acc = {}.'.format(acc*100))
    return Y_pred, Y_test, acc

def linear_clf(X_train, y_train, X_test, y_test):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred==y_test)/len(y_test)
    return y_pred, acc

def main(sub, stimuli, freq_bands, n_perm=1000):
    """
    Train on one session and test on the other session.
    """
    classes = le.transform(stimuli)

    fname_A = os.path.join(preprocessed_data_dir, 'PSD', f'sub{sub}A.npz')
    fname_B = os.path.join(preprocessed_data_dir, 'PSD', f'sub{sub}B.npz')

    if not (os.path.exists(fname_A) and os.path.exists(fname_B)):
        raise ValueError(f'Two sessions are required for sub{sub}.\n')
        return

    X_A, y_A = load_data(fname_A, classes, freq_bands, psd_res=1)
    X_B, y_B = load_data(fname_B, classes, freq_bands, psd_res=1)

    # Train on A Test on B
    _, acc_A = linear_clf(X_A, y_A, X_B, y_B)
    print(f"Acc of clf trained on A: {acc_A*100:.3f}%")

    # Train on B Test on A
    _, acc_B = linear_clf(X_B, y_B, X_A, y_A)
    print(f"Acc of clf trained on B: {acc_B*100:.3f}%")

    avg_acc = (acc_A + acc_B)/2
    print(f'Avg Acc {avg_acc*100:.3f}%')

    # Permutation test
    acc_perm = []
    for _ in tqdm(range(n_perm)):
        np.random.shuffle(y_A)
        _, tmp_acc = linear_clf(X_A, y_A, X_B, y_B)
        acc_perm.append(tmp_acc)
    p = np.sum(acc_A > np.array(acc_perm))/n_perm
    print(f'P-value trained on A: {1-p:.3f}')

    acc_perm = []
    for _ in tqdm(range(n_perm)):
        np.random.shuffle(y_B)
        _, tmp_acc = linear_clf(X_B, y_B, X_A, y_A)
        acc_perm.append(tmp_acc)

    p = np.sum(acc_B > np.array(acc_perm))/n_perm
    print(f'P-value trained on B: {1-p:.3f}')
    print('')
    return


if __name__ == '__main__':
    import argparse

    stimuli_choice = ['violetNoise', 'K448orig_120', 'classical_N', 'Wagner']

    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type=str, help='Subject ID')
    parser.add_argument('class1', type=str, choices=stimuli_choice, help='Class Label 1')
    parser.add_argument('class2', type=str, choices=stimuli_choice, help='Class Label 2')
    parser.add_argument('--test_config', help='A .json file of testing configuration.')


    args = parser.parse_args()
    if not args.test_config:
        freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'ripple']
    else:
        with open(args.test_config, 'r') as f:
            test_config = json.load(f)
        freq_bands = test_config['freq_bands']

    if args.class1 == args.class2:
        raise ValueError("Require different classe labels.")

    stimuli = [args.class1, args.class2]

    main(args.sub, stimuli, freq_bands)
