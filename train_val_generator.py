import numpy as np
from os.path import join, isfile
import os
from os import listdir

path = '/home/asad/score-based-channels/data/train-val-data'
os.chdir(path)

for i in range(1, 6):
    path = '/home/asad/score-based-channels/data/raw-range-doppler'
    filenames = np.sort([join(path,f) for f in listdir(path) if isfile(join(path, f))])
    data = []

    # 80, 20 split
    train_set = np.random.choice(filenames, size=int(len(filenames)*0.8), replace=False)
    val_set = np.array(list(set(filenames) - set(train_set)))

    np.savetxt('train-experiment-' + str(i) + '.txt', train_set, fmt='%s')
    np.savetxt('val-experiment-' + str(i) + '.txt', val_set, fmt='%s')