import os
from tqdm import tqdm
import json
from icecream import ic


def check_render(dataset, st=None, end=None):
    total_lists = []
    with open(dataset+'.json', 'r') as f:
        glb_list = json.load(f)
        for x in glb_list:
            total_lists.append(x.split('/')[-2] )
    
    if st is not None:
        end = min(end, len(total_lists))
        total_lists = total_lists[st:end]
        glb_list = glb_list[st:end]
    
    save_dir = '/data/lipeng/human_8view_with_smplx/'+dataset
    unrendered = set(total_lists) - set(os.listdir(save_dir))

    num_finish = 0
    num_failed = len(unrendered)
    failed_case = []
    for case in os.listdir(save_dir):
        if not os.path.exists(os.path.join(save_dir, case, 'smpl_normal', '007.png')):                
            failed_case.append(case)
            num_failed += 1
        else:
            num_finish += 1
    ic(num_failed)
    ic(num_finish)


    need_render = []
    for full_path in glb_list:
        for case in failed_case:
            if case in full_path:
                need_render.append(full_path)

    with open('need_render.json', 'w') as f:
        json.dump(need_render, f, indent=4)

if __name__ == '__main__':
    dataset = 'THuman2.1'
    check_render(dataset)