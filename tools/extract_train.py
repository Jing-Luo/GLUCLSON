import os
import tarfile
from tqdm import tqdm

root_folder = "ILSVRC2012_img_train"
pbar = tqdm(total=1000)
for class_tar in os.listdir(root_folder):
    class_name = class_tar.split('.')[0]
    class_dir = root_folder + '/' + class_name
    class_path = root_folder + '/' + class_tar
    pbar.set_description("Extracting " + class_name)
    with tarfile.open(class_path) as f:
        f.extractall(class_dir)
    os.remove(class_path)
    pbar.update(1)
pbar.close()


