import os
from tqdm import tqdm
from glob import glob

def rename_customhumans():
    root = '/data/lipeng/human_scan/CustomHumans/smplx/'
    file_paths = glob(os.path.join(root, '*/*_smpl.obj'))
    for file_path in tqdm(file_paths):
        new_path = file_path.replace('_smpl', '')
        os.rename(file_path, new_path)  
        
def rename_thuman21():
    root = '/data/lipeng/human_scan/THuman2.1/smplx/'
    file_paths = glob(os.path.join(root, '*/*.obj'))
    for file_path in tqdm(file_paths):
        obj_name = file_path.split('/')[-2]
        folder_name = os.path.dirname(file_path)
        new_path = os.path.join(folder_name, obj_name+'.obj')
        # print(new_path)
        # print(file_path)
        os.rename(file_path, new_path)  

if __name__ == '__main__':
    rename_thuman21()
    rename_customhumans()