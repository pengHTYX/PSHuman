import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image



def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(RT[:, :3],xyz.T).T + RT[:, 3:].T
    xyz = np.dot(K,xyz.T).T
    xy = xyz[:, :2] + 256
    return xy


def get_rays(H, W, K, R, T):
    # w2c=np.concatenate([R,T],axis=1)
    # w2c=np.concatenate([w2c,[[0,0,0,1]]],axis=0)
    # c2w=np.linalg.inv(w2c)
    # i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # dirs = np.stack([(i-256)/K[0][0], -(j-256)/K[1][1], -np.ones_like(i)], -1)
    # # Rotate ray directions from camera frame to the world frame
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    # calculate the camera origin
    rays_o = -np.dot(np.linalg.inv(R), T).ravel()+np.array([0,0,500])
    # calculate the world coordinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    #xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.stack([(i-256)/K[0][0], -(j-256)/K[1][1], -np.ones_like(i)], -1)
    pixel_world = np.dot(R.T, (pixel_camera - T.ravel()).reshape(-1,3).T).T.reshape(H,W,3)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / (ray_d[:, None] + 1e-9)).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, training = True):
    H, W = img.shape[:2]
    K[2,2]=1
    ray_o, ray_d = get_rays(H, W, K, R, T)  # world coordinate

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)  # 可视化bound mask
    # # bound_mask [512,512]
    # # save bound mask as image
    # bound_mask = bound_mask.astype(np.uint8)
    # bound_mask = bound_mask * 255
    # bound_mask = Image.fromarray(bound_mask)
    # msk_image=Image.fromarray(msk)
    # bound_mask.save('bound_mask.png')
    # msk_image.save('msk.png')


    img[bound_mask != 1] = 0

    #msk = msk * bound_mask

    
    if training:
        nsampled_rays = 0
        # face_sample_ratio = cfg.face_sample_ratio
        # body_sample_ratio = cfg.body_sample_ratio
        body_sample_ratio = 0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        body_mask_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk > 0)
            
            coord_body = coord_body[np.random.randint(0, len(coord_body)-1, n_body)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask > 0)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            body_mask_ = msk[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            body_mask_list.append(body_mask_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        body_mask = (np.concatenate(body_mask_list) > 0).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        body_mask = msk.reshape(-1).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        mask_at_box = np.logical_and(mask_at_box > 0, body_mask > 0)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        body_mask = body_mask[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, body_mask, ray_o, ray_d, near, far, coord, mask_at_box


def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(z_vals.device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = raw[...,:3]  # [N_rays, N_samples, 3]A
    noise = 0.
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]  
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(z_vals.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]  #后面的cumprod是累乘函数，是求Ti这个积分项
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]  C and c

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(z_vals.device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
    return rgb_map, disp_map, acc_map, weights, depth_map


def get_wsampling_points(ray_o, ray_d, near, far):
        """
        sample pts on rays
        """
        N_samples=64
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[ :, None] + ray_d[ :, None] * z_vals[..., None]

        return pts, z_vals
