from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import math
import cv2
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


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image