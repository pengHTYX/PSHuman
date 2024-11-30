import os
import json
def find_files(directory, extensions):
    results = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extensions):
                file_path = os.path.abspath(os.path.join(foldername, filename))
                results.append(file_path)
    return results

def count_customhumans(root):
    directory_path = ['CustomHumans/mesh']

    extensions = ('.ply', '.obj')

    lists = []
    for dataset_path in directory_path:
        dir = os.path.join(root, dataset_path)
        file_paths = find_files(dir, extensions)
        # import pdb;pdb.set_trace()
        dataset_name = dataset_path.split('/')[0]
        for file_path in file_paths:
            lists.append(file_path.replace(root, ""))
    with open(f'{dataset_name}.json', 'w') as f:
        json.dump(lists, f, indent=4)

def count_thuman21(root):
    directory_path = ['THuman2.1/mesh']
    extensions = ('.ply', '.obj')
    lists = []
    for dataset_path in directory_path:
        dir = os.path.join(root, dataset_path)
        file_paths = find_files(dir, extensions)
        dataset_name = dataset_path.split('/')[0]
        for file_path in file_paths:
            lists.append(file_path.replace(root, ""))
    with open(f'{dataset_name}.json', 'w') as f:
        json.dump(lists, f, indent=4)

if __name__ == '__main__':
    root = '/data/lipeng/human_scan/'  
    # count_customhumans(root)
    count_thuman21(root)