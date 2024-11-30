import os
from glob import glob
from rembg import remove
from argparse import ArgumentParser
from PIL import Image
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to input images')
    args = parser.parse_args()
    
    imgs = glob(os.path.join(args.path, '*.png')) + glob(os.path.join(args.path, '*.jpg'))
    for img in imgs:
        path = os.path.dirname(img)
        name = os.path.basename(img).split('.')[0] + '_rmbg.png'

        img_np = Image.open(img)        
        img = remove(img_np)
        img.save(os.path.join(args.path, name))