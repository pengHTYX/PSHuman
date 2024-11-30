import datetime
import pytz
import traceback
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import json 
import os
from tqdm import tqdm   
import cv2
import imageio
def get_time_for_log():
    return datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime(
        "%Y%m%d %H:%M:%S")


def get_trace_for_log():
    return str(traceback.format_exc())

def make_grid_(imgs, save_file, nrow=10, pad_value=1):
    if isinstance(imgs, list):
        if isinstance(imgs[0], Image.Image):
            imgs = [torch.from_numpy(np.array(img)/255.) for img in imgs]
        elif isinstance(imgs[0], np.ndarray):
            imgs = [torch.from_numpy(img/255.) for img in imgs]
        imgs = torch.stack(imgs, 0).permute(0, 3, 1, 2)
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)

    img_grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=pad_value)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype(np.uint8)
    img_grid = Image.fromarray(img_grid)
    img_grid.save(save_file) 
    
def draw_caption(img, text, pos, size=100, color=(128, 128, 128)):
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(size= size)
    font = ImageFont.load_default()
    font = font.font_variant(size=size)
    draw.text(pos, text, color, font=font)
    return img


def txt2json(txt_file, json_file):  
    with open(txt_file, 'r') as f:
        items = f.readlines()
        items = [x.strip() for x in items]

    with open(json_file, 'w') as f:
        json.dump(items.tolist(), f)
        
def process_thuman_texture():
    path = '/aifs4su/mmcode/lipeng/Thuman2.0'
    cases = os.listdir(path)
    for case in tqdm(cases):
        mtl = os.path.join(path, case, 'material0.mtl')
        with open(mtl, 'r') as f:
            lines = f.read()
            lines = lines.replace('png', 'jpeg')
        with open(mtl, 'w') as f:
            f.write(lines)
        

#### for debug
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)

def read_data(data_dir, i):
    """
    Return:
    rgb: (H, W, 3) torch.float32
    depth: (H, W, 1) torch.float32
    mask: (H, W, 1) torch.float32
    c2w: (4, 4) torch.float32
    intrinsic: (3, 3) torch.float32
    """
    background_color = torch.tensor([0.0, 0.0, 0.0])

    rgb_name = os.path.join(data_dir, f'render_%04d.webp' % i)
    depth_name = os.path.join(data_dir, f'depth_%04d.exr' % i)
    
    img = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(rgb_name))
                    .convert("RGBA")
                )
                / 255.0
            ).float()
    mask = img[:, :, -1:]
    rgb = img[:, :, :3] * mask + background_color[
        None, None, :
    ] * (1 - mask) 

    depth = torch.from_numpy(
        cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)[..., 0, None]
    )
    mask[depth > 100.0] = 0.0
    depth[~(mask > 0.5)] = 0.0  # set invalid depth to 0

    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    c2w = torch.as_tensor(
                meta['locations'][i]["transform_matrix"],
                dtype=torch.float32,
            )

    H, W = rgb.shape[:2]
    fovy = meta["camera_angle_x"]
    intrinsic = get_intrinsic_from_fov(fovy, H=H, W=W)

    return rgb, depth, mask, c2w, intrinsic
