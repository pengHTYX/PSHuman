from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from utils.func import  make_sparse_camera,  make_round_views
from utils.render import NormalsRenderer
import  torch.optim as optim
from tqdm import tqdm
from utils.video_utils import write_video
from omegaconf import OmegaConf
import numpy as np
import os
from PIL import Image
import kornia
import torch
import torch.nn as nn
import trimesh
from icecream import ic
from utils.project_mesh import multiview_color_projection, get_cameras_list
from utils.mesh_utils import  to_py3d_mesh, rot6d_to_rotmat, tensor2variable
from utils.project_mesh import  project_color, get_cameras_list
from utils.smpl_util import SMPLX
from lib.dataset.mesh_util import apply_vertex_mask, part_removal, poisson, keep_largest
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import argparse
#### ------------------- config----------------------   
bg_color = np.array([1,1,1])

class colorModel(nn.Module):
    def __init__(self, renderer, v, f, c):
        super().__init__()
        self.renderer = renderer
        self.v = v
        self.f = f
        self.colors = nn.Parameter(c, requires_grad=True)
        self.bg_color = torch.from_numpy(bg_color).float().to(self.colors.device)
    def forward(self, return_mask=False):
        rgba = self.renderer.render(self.v, self.f, colors=self.colors)
        if return_mask:
            return rgba
        else:
            mask = rgba[..., 3:]
            return rgba[..., :3] * mask + self.bg_color * (1 - mask)


def scale_mesh(vert):
    min_bbox, max_bbox = vert.min(0)[0], vert.max(0)[0]
    center = (min_bbox + max_bbox) / 2 
    offset = -center
    vert = vert + offset

    max_dist = torch.max(torch.sqrt(torch.sum(vert**2, dim=1)))
    scale = 1.0 / max_dist
    return scale, offset

def save_mesh(save_name, vertices, faces,  color=None):
    trimesh.Trimesh(
        vertices.detach().cpu().numpy(), 
        faces.detach().cpu().numpy(), 
        vertex_colors=(color.detach().cpu().numpy() * 255).astype(np.uint8) if color is not None else None) \
    .export(save_name)
        

    

class ReMesh:
    def __init__(self, opt, econ_dataset):
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.num_view = opt.num_view
        
        self.out_path = opt.res_path
        os.makedirs(self.out_path, exist_ok=True)
        self.resolution = opt.resolution
        self.views = ['front_face', 'front_right', 'right', 'back', 'left', 'front_left' ]
        self.weights = torch.Tensor([1., 0.4, 0.8, 1.0, 0.8, 0.4]).view(6,1,1,1).to(self.device)
      
        self.renderer = self.prepare_render()
        # pose prediction
        self.econ_dataset = econ_dataset
        self.smplx_face =  torch.Tensor(econ_dataset.faces.astype(np.int64)).long().to(self.device)
    
    def prepare_render(self):
        ### ------------------- prepare camera and renderer----------------------
        mv, proj = make_sparse_camera(self.opt.cam_path, self.opt.scale, views=[0,1,2,4,6,7], device=self.device)
        renderer = NormalsRenderer(mv, proj, [self.resolution, self.resolution], device=self.device)
        return renderer
    
    def proj_texture(self, fused_images, vertices, faces):
        mesh = to_py3d_mesh(vertices, faces)
        mesh = mesh.to(self.device)
        camera_focal =  1/2
        cameras_list = get_cameras_list([0, 45, 90, 180, 270, 315], device=self.device, focal=camera_focal)
        mesh = multiview_color_projection(mesh, fused_images, camera_focal=camera_focal, resolution=self.resolution, weights=self.weights.squeeze().cpu().numpy(),
                                          device=self.device, complete_unseen=True, confidence_threshold=0.2, cameras_list=cameras_list)
        return mesh
    
    def get_invisible_idx(self, imgs, vertices, faces):
        mesh = to_py3d_mesh(vertices, faces)
        mesh = mesh.to(self.device)
        camera_focal =  1/2
        if self.num_view == 6:
            cameras_list = get_cameras_list([0, 45, 90, 180, 270, 315], device=self.device, focal=camera_focal)
        elif self.num_view == 4:
            cameras_list = get_cameras_list([0, 90, 180, 270], device=self.device, focal=camera_focal)
        valid_vert_id = []
        vertices_colors = torch.zeros((vertices.shape[0], 3)).float().to(self.device)
        valid_cnt = torch.zeros((vertices.shape[0])).to(self.device)
        for  cam, img, weight in zip(cameras_list, imgs, self.weights.squeeze()):
            ret = project_color(mesh, cam, img, eps=0.01, resolution=self.resolution, device=self.device)
            # print(ret['valid_colors'].shape)
            valid_cnt[ret['valid_verts']] += weight
            vertices_colors[ret['valid_verts']] += ret['valid_colors']*weight
        valid_mask = valid_cnt > 1
        invalid_mask = valid_cnt < 1
        vertices_colors[valid_mask] /= valid_cnt[valid_mask][:, None] 
        
        # visibility
        invisible_vert = valid_cnt < 1
        invisible_vert_indices = torch.nonzero(invisible_vert).squeeze()
        # vertices_colors[invalid_vert] = torch.tensor([1.0, 0.0, 0.0]).float().to("cuda")
        return vertices_colors, invisible_vert_indices 
    
    def inpaint_missed_colors(self, all_vertices, all_colors, missing_indices):
        all_vertices = all_vertices.detach().cpu().numpy()
        all_colors = all_colors.detach().cpu().numpy()
        missing_indices = missing_indices.detach().cpu().numpy()


        non_missing_indices = np.setdiff1d(np.arange(len(all_vertices)), missing_indices)

        kdtree = KDTree(all_vertices[non_missing_indices])


        for missing_index in missing_indices:
            missing_vertex = all_vertices[missing_index]
        
            _, nearest_index = kdtree.query(missing_vertex.reshape(1, -1))
            
            interpolated_color = all_colors[non_missing_indices[nearest_index]]
            
            all_colors[missing_index] = interpolated_color
        
        return torch.from_numpy(all_colors).to(self.device)

    def load_training_data(self, case):
        ###------------------ load target images -------------------------------
        kernal = torch.ones(3, 3)
        erode_iters = 2
        normals = []
        masks = []
        colors = []
        for idx, view in enumerate(self.views):
        # for idx  in [0,2,3,4]:
            normal = Image.open(f'{self.opt.mv_path}/{case}/normals_{view}_masked.png')
            # normal = Image.open(f'{data_path}/{case}/normals/{idx:02d}_rgba.png')
            normal = normal.convert('RGBA').resize((self.resolution, self.resolution), Image.BILINEAR)
            normal = np.array(normal).astype(np.float32) / 255.
            mask = normal[..., 3:]  # alpha
            mask_troch = torch.from_numpy(mask).unsqueeze(0)
            for _ in range(erode_iters):
                mask_torch = kornia.morphology.erosion(mask_troch, kernal)
            mask_erode = mask_torch.squeeze(0).numpy()
            masks.append(mask_erode)
            normal = normal[..., :3] * mask_erode 
            normals.append(normal)
            
            color = Image.open(f'{self.opt.mv_path}/{case}/color_{view}_masked.png')
            color = color.convert('RGBA').resize((self.resolution, self.resolution), Image.BILINEAR)
            color = np.array(color).astype(np.float32) / 255.
            color_mask = color[..., 3:]  # alpha
            # color_dilate = color[..., :3] * color_mask  + bg_color * (1 - color_mask)
            color_dilate = color[..., :3] * mask_erode + bg_color * (1 - mask_erode)
            colors.append(color_dilate)

        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(self.device)
        normals = np.stack(normals, 0) 
        target_normals = torch.from_numpy(normals).to(self.device)
        colors = np.stack(colors, 0)
        target_colors = torch.from_numpy(colors).to(self.device)
        return masks, target_colors, target_normals
    
    def preprocess(self, color_pils, normal_pils):
          ###------------------ load target images -------------------------------
        kernal = torch.ones(3, 3)
        erode_iters = 2
        normals = []
        masks = []
        colors = []
        for normal, color in zip(normal_pils, color_pils):
            normal = normal.resize((self.resolution, self.resolution), Image.BILINEAR)
            normal = np.array(normal).astype(np.float32) / 255.
            mask = normal[..., 3:]  # alpha
            mask_troch = torch.from_numpy(mask).unsqueeze(0)
            for _ in range(erode_iters):
                mask_torch = kornia.morphology.erosion(mask_troch, kernal)
            mask_erode = mask_torch.squeeze(0).numpy()
            masks.append(mask_erode)
            normal = normal[..., :3] * mask_erode 
            normals.append(normal)
            
            color = color.resize((self.resolution, self.resolution), Image.BILINEAR)
            color = np.array(color).astype(np.float32) / 255.
            color_mask = color[..., 3:]  # alpha
            # color_dilate = color[..., :3] * color_mask  + bg_color * (1 - color_mask)
            color_dilate = color[..., :3] * mask_erode + bg_color * (1 - mask_erode)
            colors.append(color_dilate)

        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(self.device)
        normals = np.stack(normals, 0) 
        target_normals = torch.from_numpy(normals).to(self.device)
        colors = np.stack(colors, 0)
        target_colors = torch.from_numpy(colors).to(self.device)
        return masks, target_colors, target_normals
    
    def optimize_case(self, case, pose, clr_img, nrm_img, opti_texture=True):
        case_path = f'{self.out_path}/{case}'
        os.makedirs(case_path, exist_ok=True)
        
        if clr_img is not None:
            masks, target_colors, target_normals = self.preprocess(clr_img, nrm_img)
        else:
            masks, target_colors, target_normals = self.load_training_data(case)
        
        # rotation
        rz = R.from_euler('z', 180, degrees=True).as_matrix()
        ry = R.from_euler('y', 180, degrees=True).as_matrix()
        rz = torch.from_numpy(rz).float().to(self.device)
        ry = torch.from_numpy(ry).float().to(self.device)
        
        scale, offset = None, None

        global_orient = pose["global_orient"] # pymaf_res[idx]['smplx_params']['body_pose'][:, :1, :, :2].to(device).reshape(1, 1, -1) # data["global_orient"]
        body_pose = pose["body_pose"] # pymaf_res[idx]['smplx_params']['body_pose'][:, 1:22, :, :2].to(device).reshape(1, 21, -1) # data["body_pose"]
        left_hand_pose = pose["left_hand_pose"] # pymaf_res[idx]['smplx_params']['left_hand_pose'][:, :, :, :2].to(device).reshape(1, 15, -1)
        right_hand_pose = pose["right_hand_pose"] # pymaf_res[idx]['smplx_params']['right_hand_pose'][:, :, :, :2].to(device).reshape(1, 15, -1)
        beta = pose["betas"]
        
        # The optimizer and variables
        optimed_pose = torch.tensor(body_pose,
                                device=self.device,
                                requires_grad=True)  # [1,23,3,3]
        optimed_trans = torch.tensor(pose["trans"],
                                        device=self.device,
                                        requires_grad=True)  # [3]
        optimed_betas = torch.tensor(beta,
                                        device=self.device,
                                        requires_grad=True)  # [1,200]
        optimed_orient = torch.tensor(global_orient,
                                        device=self.device,
                                        requires_grad=True)  # [1,1,3,3]
        optimed_rhand = torch.tensor(right_hand_pose,
                                        device=self.device,
                                        requires_grad=True)  
        optimed_lhand = torch.tensor(left_hand_pose,
                                        device=self.device,
                                        requires_grad=True)  
        
        optimed_params = [
            {'params': [optimed_lhand, optimed_rhand], 'lr': 1e-3},
            {'params': [optimed_betas, optimed_trans, optimed_orient, optimed_pose], 'lr': 3e-3},
        ]
        optimizer_smpl = torch.optim.Adam(
            optimed_params,
            amsgrad=True,
        )
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=5,
        )
        smpl_steps = 100
        
        for i in tqdm(range(smpl_steps)):
            optimizer_smpl.zero_grad()
            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(
                -1, 6)).unsqueeze(0)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(
                -1, 6)).unsqueeze(0)

            smpl_verts, smpl_landmarks, smpl_joints = self.econ_dataset.smpl_model(
                shape_params=optimed_betas,
                expression_params=tensor2variable(pose["exp"], self.device),
                body_pose=optimed_pose_mat,
                global_pose=optimed_orient_mat,
                jaw_pose=tensor2variable(pose["jaw_pose"], self.device),
                left_hand_pose=optimed_lhand,
                right_hand_pose=optimed_rhand,

            )

            smpl_verts = smpl_verts + optimed_trans
            
            v_smpl = torch.matmul(torch.matmul(smpl_verts.squeeze(0), rz.T), ry.T)
            if scale is None:
                scale, offset = scale_mesh(v_smpl.detach())
            v_smpl = (v_smpl + offset) * scale * 2
            # if i == 0:
            #   save_mesh(f'{case_path}/{case}_init_smpl.obj', v_smpl, self.smplx_face)
            # exit()
            normals = calc_vertex_normals(v_smpl, self.smplx_face)
            nrm = self.renderer.render(v_smpl, self.smplx_face, normals=normals)

            masks_ = nrm[..., 3:] 
            smpl_mask_loss = ((masks_ - masks) * self.weights).abs().mean()  
            smpl_nrm_loss = ((nrm[..., :3] - target_normals) * self.weights).abs().mean()

            smpl_loss =  smpl_mask_loss + smpl_nrm_loss
            # smpl_loss =  smpl_mask_loss 
            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)

        mesh_smpl = trimesh.Trimesh(vertices=v_smpl.detach().cpu().numpy(), faces=self.smplx_face.detach().cpu().numpy())  

      
        nrm_opt = MeshOptimizer(v_smpl.detach(), self.smplx_face.detach(), edge_len_lims=[0.01, 0.1])
        vertices, faces = nrm_opt.vertices, nrm_opt.faces
        
        # ###----------------------- optimization iterations-------------------------------------
        for i in tqdm(range(self.opt.iters)):
            nrm_opt.zero_grad()

            normals = calc_vertex_normals(vertices,faces)
            nrm = self.renderer.render(vertices,faces, normals=normals)
            normals = nrm[..., :3]   
            # if i < 800:
            loss = ((normals-target_normals) * self.weights).abs().mean()
            # else:
            #     loss = ((normals-target_images) * masks).abs().mean()
            
            alpha = nrm[..., 3:]
            loss += ((alpha - masks) * self.weights).abs().mean()

            loss.backward()
            
            nrm_opt.step()
            
            vertices,faces = nrm_opt.remesh()
            
            if self.opt.debug and i % self.opt.snapshot_step == 0:
                import imageio
                os.makedirs(f'{case_path}/normals', exist_ok=True)
                imageio.imwrite(f'{case_path}/normals/{i:02d}.png',(nrm.detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
                # mesh_remeshed = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                # mesh_remeshed.export(f'{case_path}/{case}_remeshed_step{i}.obj')
            torch.cuda.empty_cache() 
            
        mesh_remeshed = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
        mesh_remeshed.export(f'{case_path}/{case}_remeshed.obj')
        # save_mesh(case, vertices, faces)
        vertices = vertices.detach()
        faces = faces.detach()

        #### replace hand
        smpl_data = SMPLX()
        if self.opt.replace_hand  and True in pose['hands_visibility'][0]:
            hand_mask = torch.zeros(smpl_data.smplx_verts.shape[0], )
            if pose['hands_visibility'][0][0]:
                hand_mask.index_fill_(
                    0, torch.tensor(smpl_data.smplx_mano_vid_dict["left_hand"]), 1.0
                )
            if pose['hands_visibility'][0][1]:
                hand_mask.index_fill_(
                    0, torch.tensor(smpl_data.smplx_mano_vid_dict["right_hand"]), 1.0
                )

            hand_mesh = apply_vertex_mask(mesh_smpl.copy(), hand_mask)
            body_mesh = part_removal(
                mesh_remeshed.copy(),
                hand_mesh,
                0.08,
                self.device,
                mesh_smpl.copy(),
                region="hand"
            )
            final = poisson(sum([hand_mesh, body_mesh]), f'{case_path}/{case}_final.obj', 10, False)
        else:
            final = poisson(mesh_remeshed, f'{case_path}/{case}_final.obj', 10, False)
        vertices = torch.from_numpy(final.vertices).float().to(self.device)
        faces = torch.from_numpy(final.faces).long().to(self.device)
        # Differing from paper, we use the texturing method in Unique3D
        masked_color = []
        for tmp in clr_img:
            # tmp = Image.open(f'{self.opt.mv_path}/{case}/color_{view}_masked.png')
            tmp = tmp.resize((self.resolution, self.resolution), Image.BILINEAR)
            tmp = np.array(tmp).astype(np.float32) / 255.
            masked_color.append(torch.from_numpy(tmp).permute(2, 0, 1).to(self.device))

        meshes = self.proj_texture(masked_color, vertices, faces)
        vertices = meshes.verts_packed().float()
        faces = meshes.faces_packed().long()
        colors = meshes.textures.verts_features_packed().float()
        save_mesh(f'./{case_path}/result_clr_scale{self.opt.scale}_{case}.obj', vertices, faces, colors)
        self.evaluate(vertices, colors, faces,  save_path=f'{case_path}/result_clr_scale{self.opt.scale}_{case}.mp4', save_nrm=True)
        

    def evaluate(self, target_vertices, target_colors, target_faces, save_path=None, save_nrm=False):
        mv, proj = make_round_views(60, self.opt.scale, device=self.device)
        renderer = NormalsRenderer(mv, proj, [512, 512], device=self.device)
        
        target_images = renderer.render(target_vertices,target_faces, colors=target_colors)
        target_images = target_images.detach().cpu().numpy()
        target_images = target_images[..., :3] * target_images[..., 3:4]  + bg_color * (1 - target_images[..., 3:4])
        target_images = (target_images.clip(0, 1) * 255).astype(np.uint8)
        
        if save_nrm:
            target_normals = calc_vertex_normals(target_vertices, target_faces)
            # target_normals[:, 2] *= -1
            target_normals = renderer.render(target_vertices, target_faces, normals=target_normals)
            target_normals = target_normals.detach().cpu().numpy()
            target_normals = target_normals[..., :3] * target_normals[..., 3:4]  + bg_color * (1 - target_normals[..., 3:4])
            target_normals = (target_normals.clip(0, 1) * 255).astype(np.uint8)
            frames = [np.concatenate([img, nrm], 1) for img, nrm in zip(target_images, target_normals)]
        else:
            frames = [img for img in target_images]
        if save_path is not None:
            write_video(frames, fps=25, save_path=save_path)
        return frames
    
    def run(self):
        cases = sorted(os.listdir(self.opt.imgs_path))  
        for idx in range(len(cases)):
            case = cases[idx].split('.')[0]
            print(f'Processing {case}')
            pose = self.econ_dataset.__getitem__(idx)
            v, f, c =  self.optimize_case(case, pose, None, None, opti_texture=True)
            self.evaluate(v, c, f,  save_path=f'{self.opt.res_path}/{case}/result_clr_scale{self.opt.scale}_{case}.mp4', save_nrm=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  help="path to the yaml configs file", default='config.yaml')
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    from econdataset import SMPLDataset
    dataset_param = {'image_dir': opt.imgs_path, 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')
    EHuman = ReMesh(opt, econdata)
    EHuman.run()

   
    
