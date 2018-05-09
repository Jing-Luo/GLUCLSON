# --------------------------------------------------------
# Preprocess imagenet validation dataset
# Create a subdir for every label, and put a symbol link
# to the image in the folder.
# --------------------------------------------------------

import shutil
import os
import pandas as pd


val_path = 'val'
image_path = '/exp/Dataset/ImageNet2012Val/ILSVRC2012_img_val' #use absolute address
synsets_file = open('../basics/synsets.txt', 'r')
val_file = open('../basics/val.txt', 'r')

if os.path.exists(val_path):
    shutil.rmtree(val_path)

synsets = [line.rstrip('\n') for line in synsets_file.readlines()]

for line in val_file.readlines():
    fname, idx = line.split()
    label_path = '%s/%s' % (val_path, synsets[int(idx)])
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    os.symlink('%s/%s' % (image_path, fname), '%s/%s' % (label_path, fname))

