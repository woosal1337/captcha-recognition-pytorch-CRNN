import os
import glob
import torch
import numpy as np

from sklearn import preprocessing, model_selection, metrics

import config
import dataset

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]
    
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1
    print(targets_enc)
    print(len(lbl_enc.classes_))

    


if __name__ == '__main__':
    run_training()