import os
import pickle
from sklearn import preprocessing

import sys
sys.path.append("../config")
from config import meta_dir, root_dir
from utils import log_event_parser

if __name__ == '__main__':
    
    stimulus_info = log_event_parser(os.path.join(meta_dir, 'sub1A.log'))
    event_label = [i.split('.')[0] for i in stimulus_info['Event']]

    le = preprocessing.LabelEncoder()
    le.fit(event_label)

    fname = os.path.join(root_dir, "classification", "event_label.pkl")
    with open(fname, 'wb') as f:
        pickle.dump(le, f)