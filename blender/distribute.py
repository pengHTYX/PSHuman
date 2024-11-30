import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os

import boto3


from glob import glob

import argparse

parser = argparse.ArgumentParser(description='distributed rendering')

parser.add_argument('--workers_per_gpu', type=int, default=10,
                    help='number of workers per gpu.')
parser.add_argument('--input_models_path', type=str, default='/data/lipeng/human_scan/',
                    help='Path to a json file containing a list of 3D object files.')
parser.add_argument('--num_gpus', type=int, default=-1,
                    help='number of gpus to use. -1 means all available gpus.')
parser.add_argument('--gpu_list',nargs='+', type=int, 
                    help='the avalaible gpus')
parser.add_argument('--resolution', type=int, default=512,
                    help='')
parser.add_argument('--random_images', type=int, default=0)
parser.add_argument('--start_i', type=int, default=0,
                    help='the index of first object to be rendered.')
parser.add_argument('--end_i', type=int, default=-1,
                    help='the index of the last object to be rendered.')

parser.add_argument('--data_dir', type=str, default='/data/lipeng/human_scan/',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--json_path', type=str, default='2K2K.json')

parser.add_argument('--save_dir', type=str, default='/data/lipeng/human_8view',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--ortho_scale', type=float, default=1.,
                    help='ortho rendering usage; how large the object is')


args = parser.parse_args()

def parse_obj_list(xs):
    cases = []
    # print(xs[:2])

    for x in xs:
        if 'THuman3.0' in x:
            # print(apath)
            splits = x.split('/')
            x = os.path.join('THuman3.0', splits[-2])
        elif 'THuman2.1' in x:
            splits = x.split('/')
            x = os.path.join('THuman2.1', splits[-2])
        elif 'CustomHumans' in x:
            splits = x.split('/')
            x = os.path.join('CustomHumans', splits[-2])
        elif '1M' in x:
            splits = x.split('/')
            x = os.path.join('2K2K', splits[-2])
        elif 'realistic_8k_model' in x:
            splits = x.split('/')
            x = os.path.join('realistic_8k_model', splits[-1].split('.')[0])
        cases.append(f'{args.save_dir}/{x}') 
    return  cases
     

with open(args.json_path, 'r') as f:
    glb_list = json.load(f)

# glb_list = ['THuman2.1/mesh/1618/1618.obj']
# glb_list = ['THuman3.0/00024_1/00024_0006/mesh.obj']
# glb_list = ['CustomHumans/mesh/0383_00070_02_00061/mesh-f00061.obj'] 
# glb_list = ['1M/01968/01968.ply', '1M/00103/00103.ply']
# glb_list = ['realistic_8k_model/01aab099a2fe4af7be120110a385105d.glb']

total_num_glbs = len(glb_list)



def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    print("Worker started")
    while True:
        case, save_p = queue.get()
        src_path = os.path.join(args.data_dir, case)
        smpl_path = src_path.replace('mesh', 'smplx', 1)
        
        command = ('blender -b -P blender_render_human_ortho.py'
        f' -- --object_path {src_path}'
        f' --smpl_path {smpl_path}'
        f' --output_dir {save_p} --engine CYCLES'
        f' --resolution {args.resolution}'
        f' --random_images {args.random_images}'
        )
       
        print(command)
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()
    

if __name__ == "__main__":
    # args = tyro.cli(Args)

    s3 = None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, args.gpu_list[gpu_i], s3)
            )
            process.daemon = True
            process.start()
        
    # Add items to the queue
    
    save_dirs = parse_obj_list(glb_list)
    args.end_i = len(save_dirs) if args.end_i > len(save_dirs) or args.end_i==-1 else args.end_i
    
    for case_sub, save_dir in zip(glb_list[args.start_i:args.end_i], save_dirs[args.start_i:args.end_i]):
        queue.put([case_sub, save_dir])

   

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
