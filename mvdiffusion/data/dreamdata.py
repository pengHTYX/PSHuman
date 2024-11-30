import numpy as np
import torch
from torch.utils.data import Dataset
import json
from typing import Tuple, Optional, Any
import cv2
import random
import os
import math
from PIL import Image, ImageOps
from .normal_utils import  worldNormal2camNormal, img2normal, norm_normalize
from icecream import ic
def shift_list(lst, n):
    length = len(lst)
    n = n % length  # Ensure n is within the range of the list length
    return lst[-n:] + lst[:-n]


class ObjaverseDataset(Dataset):
    def __init__(self,
        root_dir: str,
        azi_interval: float,
        random_views: int,
        predict_relative_views: list,
        bg_color: Any,
        object_list: str,
        prompt_embeds_path: str,
        img_wh: Tuple[int, int],
        validation: bool = False,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        # augment_data: bool = False,
        side_views_rate: float = 0.,
        read_normal: bool = True,
        read_color: bool = False,
        read_depth: bool = False,
        mix_color_normal: bool = False,
        random_view_and_domain: bool = False,
        load_cache: bool = False,
        exten: str = '.png',
        elevation_list: Optional[str] = None,
        with_smpl: Optional[bool] = False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.fixed_views = int(360 // azi_interval)
        self.bg_color = bg_color
        self.validation = validation
        self.num_samples = num_samples
        self.trans_norm_system = trans_norm_system
        # self.augment_data = augment_data
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.mix_color_normal = mix_color_normal  # mix load color and normal maps
        self.random_view_and_domain = random_view_and_domain # load normal or rgb of a single view
        self.random_views = random_views
        self.load_cache = load_cache
        self.total_views = int(self.fixed_views * (self.random_views + 1))
        self.predict_relative_views = predict_relative_views
        self.pred_view_nums = len(self.predict_relative_views)
        self.exten = exten
        self.side_views_rate = side_views_rate
        self.with_smpl = with_smpl
        if self.with_smpl:
            self.smpl_image_path = 'smpl_image'
            self.smpl_normal_path = 'smpl_normal'
            
        
        ic(self.total_views)
        ic(self.fixed_views)
        ic(self.predict_relative_views)
        ic(self.with_smpl)
        
        self.objects = []
        if object_list is not None:
            for dataset_list in object_list:
                with open(dataset_list, 'r') as f:
                    objects = json.load(f)
                self.objects.extend(objects)
        else:
            self.objects = os.listdir(self.root_dir)

        # load fixed camera poses
        self.trans_cv2gl_mat = np.linalg.inv(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        self.fix_cam_poses = []
        camera_path = os.path.join(self.root_dir, self.objects[0], 'camera')
        for vid in range(0, self.total_views, self.random_views+1):
            cam_info = np.load(f'{camera_path}/{vid:03d}.npy', allow_pickle=True).item()
            assert cam_info['camera'] == 'ortho', 'Only support predict ortho camera !!!'
            self.fix_cam_poses.append(cam_info['extrinsic'])
        random.shuffle(self.objects)
        
        
        if elevation_list:
            with open(elevation_list, 'r') as f:
                ele_list = [o.strip() for o in f.readlines()] 
            self.objects = set(ele_list) & set(self.objects)
          
        self.all_objects = set(self.objects)
        self.all_objects = list(self.all_objects)
        
        self.validation = validation
        if not validation:
            self.all_objects = self.all_objects[:-num_validation_samples]
            # print('Warning: you are fitting in small-scale dataset')
            # self.all_objects = self.all_objects
        else:
            self.all_objects = self.all_objects[-num_validation_samples:]
            
        if num_samples is not None:
            self.all_objects = self.all_objects[:num_samples]
        ic(len(self.all_objects))
        print(f"loaded {len(self.all_objects)} in the dataset")
        
        normal_prompt_embedding = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
        color_prompt_embedding = torch.load(f'{prompt_embeds_path}/clr_embeds.pt')
        if len(self.predict_relative_views) == 6:
            self.normal_prompt_embedding = normal_prompt_embedding
            self.color_prompt_embedding = color_prompt_embedding
        elif len(self.predict_relative_views) == 4:
            self.normal_prompt_embedding = torch.stack([normal_prompt_embedding[0], normal_prompt_embedding[2], normal_prompt_embedding[3], normal_prompt_embedding[4], normal_prompt_embedding[6]] , 0)
            self.color_prompt_embedding = torch.stack([color_prompt_embedding[0], color_prompt_embedding[2], color_prompt_embedding[3], color_prompt_embedding[4], color_prompt_embedding[6]] , 0)

        # flip back and left views
        if len(self.predict_relative_views) == 6:
            self.flip_views = [3, 4]  
        elif len(self.predict_relative_views) == 4:
            self.flip_views = [2, 3]  
        
        # self.backup_data = self.__getitem_norm__(0, 'Thuman2.0/0340') 
        self.backup_data = self.__getitem_norm__(0)
    
    def trans_cv2gl(self, rt):
        r, t = rt[:3, :3], rt[:3, -1]
        r = np.matmul(self.trans_cv2gl_mat, r)   
        t = np.matmul(self.trans_cv2gl_mat, t)
        return np.concatenate([r, t[:, None]], axis=-1)
    
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:3, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:3, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def crop_image(self, top_left, img):
        size = max(self.img_wh)
        tar_size = size - top_left * 2
        
        alpha_np = np.asarray(img)[:, :, 3]
        
        
        coords = np.argwhere(alpha_np > 0.5)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
    
        img = img.crop((x_min, y_min, x_max, y_max)).resize((tar_size, tar_size))
        img = ImageOps.expand(img, border=(top_left, top_left, top_left, top_left), fill=0)
        return img
    
    def load_cropped_img(self, img_path, bg_color, top_left, return_type='np'):
        rgba = Image.open(img_path)
        rgba = self.crop_image(top_left, rgba)
        rgba = np.array(rgba)
        rgba = rgba.astype(np.float32) / 255. # [0, 1]
        img, alpha = rgba[..., :3], rgba[..., 3:4]
        
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
        
            
    def load_image(self, img_path, bg_color, alpha=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        rgba = np.array(Image.open(img_path).resize(self.img_wh))
        rgba = rgba.astype(np.float32) / 255. # [0, 1]
        
        img = rgba[..., :3]
        if alpha is None:
            assert rgba.shape[-1] == 4 
            alpha = rgba[..., 3:4]
        assert alpha.sum() > 1e-8, 'w/o foreground'
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
    
    
    def load_normal(self, img_path, bg_color, alpha,  RT_w2c_cond=None, return_type='np'):
        normal_np = np.array(Image.open(img_path).resize(self.img_wh))[:, :, :3]
        assert np.var(normal_np) > 1e-8, 'pure normal'
        normal_cv = img2normal(normal_np)
        
        normal_relative_cv = worldNormal2camNormal(RT_w2c_cond[:3, :3], normal_cv)
        normal_relative_cv = norm_normalize(normal_relative_cv)

        normal_relative_gl = normal_relative_cv
        normal_relative_gl[..., 1:] = -normal_relative_gl[..., 1:]

        img = (normal_relative_cv*0.5 + 0.5).astype(np.float32)  # [0, 1]
        
        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def load_halfbody_normal(self, img_path, bg_color, alpha,  RT_w2c_cond=None, return_type='np'):
        normal_np = np.array(Image.open(img_path).resize(self.img_wh).crop((256, 0, 512, 256)).resize(self.img_wh))[:, :, :3]
        assert np.var(normal_np) > 1e-8, 'pure normal'
        normal_cv = img2normal(normal_np)
        
        normal_relative_cv = worldNormal2camNormal(RT_w2c_cond[:3, :3], normal_cv)
        normal_relative_cv = norm_normalize(normal_relative_cv)
        # normal_relative_gl = normal_relative_cv[..., [ 0, 2, 1]]
        # normal_relative_gl[..., 2] = -normal_relative_gl[..., 2]
        normal_relative_gl = normal_relative_cv
        normal_relative_gl[..., 1:] = -normal_relative_gl[..., 1:]

        img = (normal_relative_cv*0.5 + 0.5).astype(np.float32)  # [0, 1]
        
        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img
    
    def __len__(self):
        return len(self.all_objects)
    
    def load_halfbody_image(self, img_path, bg_color, alpha=None, return_type='np'):

        
        rgba = np.array(Image.open(img_path).resize(self.img_wh).crop((256, 0, 512, 256)).resize(self.img_wh))
        rgba = rgba.astype(np.float32) / 255. # [0, 1]
        
        img = rgba[..., :3]
        if alpha is None:
            assert rgba.shape[-1] == 4 
            alpha = rgba[..., 3:4]
        assert alpha.sum() > 1e-8, 'w/o foreground'
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
        
    def __getitem_norm__(self, index, debug_object=None):
        # get the bg color
        bg_color = self.get_bg_color()
        if debug_object is not  None:
            object_name =  debug_object
        else:
            object_name = self.all_objects[index % len(self.all_objects)]
        face_info = np.load(f'{self.root_dir}/{object_name}/face_info.npy', allow_pickle=True).item()
        # front_fixed_idx = face_info['top3_vid'][0] // (self.random_views+1)
        if self.side_views_rate > 0 and random.random() < self.side_views_rate:
            front_fixed_idx = random.choice(face_info['top3_vid']) 
        else:
            front_fixed_idx = face_info['top3_vid'][0]
        with_face_idx = list(face_info.keys())
        with_face_idx.remove('top3_vid')
       
        assert front_fixed_idx in with_face_idx, 'not detected face'
        
        if self.validation:
            cond_ele0_idx = front_fixed_idx
            cond_random_idx = 0
        else:
            if object_name[:9] == 'realistic': # This dataset set has random pose 
                cond_ele0_idx =  random.choice(range(self.fixed_views))
                cond_random_idx = random.choice(range(self.random_views+1))
            else:
                cond_vid = front_fixed_idx
                cond_ele0_idx = cond_vid // (self.random_views + 1)
                cond_ele0_vid = cond_ele0_idx * (self.random_views + 1)
                cond_random_idx = 0
        
        # condition info
        cond_ele0_vid = cond_ele0_idx * (self.random_views + 1)
        cond_vid = cond_ele0_vid + cond_random_idx   
        cond_ele0_w2c = self.fix_cam_poses[cond_ele0_idx]
        
        img_tensors_in = [
            self.load_image(f"{self.root_dir}/{object_name}/image/{cond_vid:03d}{self.exten}", bg_color, return_type='pt')[0].permute(2, 0, 1)
        ] * self.pred_view_nums + [
            self.load_halfbody_image(f"{self.root_dir}/{object_name}/image/{cond_vid:03d}{self.exten}", bg_color, return_type='pt')[0].permute(2, 0, 1)
        ] 
        
        # output info
        pred_vids = [(cond_ele0_vid + i * (self.random_views+1)) % self.total_views  for i in self.predict_relative_views]
        # pred_w2cs = [self.fix_cam_poses[(cond_ele0_idx + i) % self.fixed_views] for i in self.predict_relative_views]
        img_tensors_out = []
        normal_tensors_out = []
        smpl_tensors_in = []
        for i, vid in enumerate(pred_vids):
            # output image
            img_tensor, alpha_ = self.load_image(f"{self.root_dir}/{object_name}/image/{vid:03d}{self.exten}", bg_color, return_type='pt')
            img_tensor = img_tensor.permute(2, 0, 1) # (3, H, W)
            if i in self.flip_views: img_tensor = torch.flip(img_tensor, [2])
            img_tensors_out.append(img_tensor)
            
            # output normal
            normal_tensor = self.load_normal(f"{self.root_dir}/{object_name}/normal/{vid:03d}{self.exten}", bg_color, alpha_.numpy(), RT_w2c_cond=cond_ele0_w2c[:3, :], return_type="pt").permute(2, 0, 1)
            if i in self.flip_views: normal_tensor = torch.flip(normal_tensor, [2]) 
            normal_tensors_out.append(normal_tensor)
            
            # input smpl image
            if self.with_smpl:
                smpl_image_tensor, smpl_alpha_ = self.load_image(f"{self.root_dir}/{object_name}/{self.smpl_image_path}/{vid:03d}{self.exten}", bg_color, return_type='pt') 
                smpl_image_tensor = smpl_image_tensor.permute(2, 0, 1) # (3, H, W)
                if i in self.flip_views: smpl_image_tensor = torch.flip(smpl_image_tensor, [2])
                smpl_tensors_in.append(smpl_image_tensor)
            
            # faces
            if i == 0:
                face_clr_out, face_alpha_out  = self.load_halfbody_image(f"{self.root_dir}/{object_name}/image/{vid:03d}{self.exten}", bg_color, return_type='pt')
                face_clr_out = face_clr_out.permute(2, 0, 1)
                face_nrm_out = self.load_halfbody_normal(f"{self.root_dir}/{object_name}/normal/{vid:03d}{self.exten}", bg_color, face_alpha_out.numpy(), RT_w2c_cond=cond_ele0_w2c[:3, :], return_type="pt").permute(2, 0, 1)
                if self.with_smpl:
                    face_smpl_in = self.load_halfbody_image(f"{self.root_dir}/{object_name}/{self.smpl_image_path}/{vid:03d}{self.exten}", bg_color, return_type='pt')[0].permute(2, 0, 1)
                
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out.append(face_clr_out)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_out.append(face_nrm_out)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)
        
        if self.with_smpl:
            smpl_tensors_in = smpl_tensors_in + [face_smpl_in]
            smpl_tensors_in = torch.stack(smpl_tensors_in, dim=0).float() # (Nv, 3, H, W)
        
        item = {            
                'id': object_name.replace('/', '_'),    
                'vid':cond_vid,
                'imgs_in': img_tensors_in,
                'imgs_out': img_tensors_out,
                'normals_out': normal_tensors_out,
                'normal_prompt_embeddings': self.normal_prompt_embedding,
                'color_prompt_embeddings': self.color_prompt_embedding,  
            }
        if self.with_smpl:
            item.update({'smpl_imgs_in': smpl_tensors_in})
        return item
    
    def __getitem__(self, index):
        try:
            data = self.__getitem_norm__(index)
            return data
        except:
            print("load error ", self.all_objects[index%len(self.all_objects)] )
            return self.backup_data


def draw_kps(image, kps):
    nose_pos = kps[2].astype(np.int32)
    top_left = nose_pos - 64
    bottom_right = nose_pos + 64
    image_cv = image.copy()
    img = cv2.rectangle(image_cv, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2)  
    return  img

if __name__ == "__main__":
    # pass
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from PIL import ImageDraw, ImageFont
    def draw_text(img, text, pos, color=(128, 128, 128)):
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(size= size)
        font = ImageFont.load_default()
        font = font.font_variant(size=10)
        draw.text(pos, text, color, font=font)
        return img
    random.seed(11)
    train_params = dict(       
        root_dir='/aifs4su/mmcode/lipeng/human_8view_with_smplx/',
        azi_interval=45.,
        random_views=0,
        predict_relative_views=[0,2,4,6],
        bg_color='white',
        object_list=['../../data_lists/human_only_scan_with_smplx.json'],
        img_wh=(768, 768),
        validation=False,
        num_validation_samples=10,
        read_normal=True,
        read_color=True,
        read_depth=False,
        # mix_color_normal=  True,
        random_view_and_domain=False,
        load_cache=False,
        exten='.png',
        prompt_embeds_path='fixed_prompt_embeds_7view',
        side_views_rate=0.1,
        with_smpl=True
    )
    train_dataset = ObjaverseDataset(**train_params)
    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    if False:
        case = 'CustomHumans/0593_00083_06_00101'
        batch = train_dataset.__getitem_norm__(0, case)
        imgs = []
        obj_name = batch['id'][:8]  
        imgs_in = batch['imgs_in']
        imgs_out = batch['imgs_out']
        normal_out = batch['normals_out']
        imgs_vis = torch.cat([imgs_in[0:1], imgs_in[-1:], imgs_out, normal_out], 0)
        img_vis = make_grid(imgs_vis, nrow=16).permute(1, 2,0)
        img_vis = (img_vis.numpy() * 255).astype(np.uint8)
        img_vis = Image.fromarray(img_vis)
        img_vis = draw_text(img_vis, obj_name, (5, 1))
        img_vis = torch.from_numpy(np.array(img_vis)).permute(2, 0, 1) / 255.
        imgs.append(img_vis)
        imgs = torch.stack(imgs, dim=0)
        img_grid = make_grid(imgs, nrow=4, padding=0)
        img_grid = img_grid.permute(1, 2, 0).numpy()
        img_grid = (img_grid * 255).astype(np.uint8)
        img_grid = Image.fromarray(img_grid)
        img_grid.save(f'../../debug/{case.replace("/", "_")}.png')
    else:
        imgs = []
        i = 0
        for batch in data_loader:
            # print(i)
            if i < 4:
                i += 1 
                obj_name = batch['id'][0][:8]  
                imgs_in = batch['imgs_in'].squeeze(0)
                smpl_in = batch['smpl_imgs_in'].squeeze(0)
                imgs_out = batch['imgs_out'].squeeze(0)
                normal_out = batch['normals_out'].squeeze(0)
                imgs_vis = torch.cat([imgs_in[0:1], imgs_in[-1:], smpl_in, imgs_out, normal_out], 0)
                img_vis = make_grid(imgs_vis, nrow=12).permute(1, 2,0)
                img_vis = (img_vis.numpy() * 255).astype(np.uint8)
                print(img_vis.shape)
                # import pdb;pdb.set_trace()
                # nose_kps = batch['face_kps'][0].numpy() 
                # print(nose_kps)
                # img_vis = draw_kps(img_vis, nose_kps)
                img_vis = Image.fromarray(img_vis)
                img_vis = draw_text(img_vis, obj_name, (5, 1))
                img_vis = torch.from_numpy(np.array(img_vis)).permute(2, 0, 1) / 255.
                imgs.append(img_vis)
            else:
                break
        imgs = torch.stack(imgs, dim=0)
        img_grid = make_grid(imgs, nrow=1, padding=0)
        img_grid = img_grid.permute(1, 2, 0).numpy()
        img_grid = (img_grid * 255).astype(np.uint8)
        img_grid = Image.fromarray(img_grid)
        img_grid.save('../../debug/noele_imgs_out_10.png') 
        

    
        
       
