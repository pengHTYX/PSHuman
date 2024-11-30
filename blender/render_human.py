import os
import json
import math
from concurrent.futures import ProcessPoolExecutor
import threading
from tqdm import tqdm

# from glcontext import egl
# egl.create_context()
# exit(0)

LOCAL_RANK = 0

num_processes = 4
NODE_RANK = int(os.getenv("SLURM_PROCID"))
WORLD_SIZE = 1
NODE_NUM=1
# NODE_RANK = int(os.getenv("SLURM_NODEID"))
IS_MAIN = False
if NODE_RANK == 0 and LOCAL_RANK == 0:
    IS_MAIN = True

GLOBAL_RANK = NODE_RANK * (WORLD_SIZE//NODE_NUM) + LOCAL_RANK


# json_path = "object_lists/Thuman2.0.json"
# json_path = "object_lists/THuman3.0.json"
json_path = "object_lists/CustomHumans.json"
data_dir = '/aifs4su/mmcode/lipeng'
save_dir = '/aifs4su/mmcode/lipeng/human_8view_new'
def parse_obj_list(x):
    if 'THuman3.0' in x:
        # print(apath)
        splits = x.split('/')
        x = os.path.join('THuman3.0', splits[-2])
    elif 'Thuman2.0' in x:
        splits = x.split('/')
        x = os.path.join('Thuman2.0', splits[-2])
    elif 'CustomHumans' in x:
        splits = x.split('/')
        x = os.path.join('CustomHumans', splits[-2])
        # print(splits[-2])
    elif '1M' in x:
        splits = x.split('/')
        x = os.path.join('2K2K', splits[-2])
    elif 'realistic_8k_model' in x:
        splits = x.split('/')
        x = os.path.join('realistic_8k_model', splits[-1].split('.')[0])
    return f'{save_dir}/{x}'  

with open(json_path, 'r') as f:
    glb_list = json.load(f)

# glb_list = ['Thuman2.0/0011/0011.obj']
# glb_list = ['THuman3.0/00024_1/00024_0006/mesh.obj']
# glb_list = ['CustomHumans/mesh/0383_00070_02_00061/mesh-f00061.obj']
# glb_list = ['realistic_8k_model/1d41f2a72f994306b80e632f1cc8233f.glb']

total_num_glbs = len(glb_list)

num_glbs_local = int(math.ceil(total_num_glbs / WORLD_SIZE))
start_idx = GLOBAL_RANK * num_glbs_local
end_idx = start_idx + num_glbs_local
# print(start_idx, end_idx)
local_glbs = glb_list[start_idx:end_idx]
if IS_MAIN:
    pbar = tqdm(total=len(local_glbs))
    lock = threading.Lock()

def process_human(glb_path):
    src_path = os.path.join(data_dir, glb_path)
    save_path = parse_obj_list(glb_path)
    # print(save_path)
    command = ('blender -b -P blender_render_human_script.py'
        f' -- --object_path {src_path}'
        f' --output_dir {save_path} ')
        # 1>/dev/null
    # print(command)
    os.system(command)

    if IS_MAIN:
        with lock:
            pbar.update(1)

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    executor.map(process_human, local_glbs)


