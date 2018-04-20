# --------------------------------------------------------
# To preprocess imagenet val dataset
# Create a subfolder for every label, and put a symbol
# link to the image in the folder.
# --------------------------------------------------------	

import shutil
import os
import pandas as pd

df = pd.read_csv('val.txt')
path = 'val'

if os.path.exists(path):
    shutil.rmtree(path)

for i, h in df.iterrows():
    fname, label = h[0].split()
    path2 = '%s/%s' % (path, label)
    if not os.path.exists(path2):
        os.makedirs(path2)
    os.symlink('ILSVRC2012_img_val/%s' % fname, '%s/%s' % (path2, fname))
