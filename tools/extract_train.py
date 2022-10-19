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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, class_dir)
    os.remove(class_path)
    pbar.update(1)
pbar.close()


