import argparse
import os
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.utils.checkpoint
from torchvision.utils import make_grid, save_image
from accelerate.utils import  set_seed
from tqdm.auto import tqdm
import torch.nn.functional as F
from einops import rearrange
from rembg import remove, new_session
import pdb
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from econdataset import SMPLDataset
from reconstruct import ReMesh
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
]
session = new_session(providers=providers)

weight_dtype = torch.float16
def tensor_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    # save_single_views: bool
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: float
    validation_grid_nrow: int

    num_views: int
    enable_xformers_memory_efficient_attention: bool
    with_smpl: Optional[bool]
    
    recon_opt: Dict


def convert_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def convert_to_pil(tensor):
    return Image.fromarray(convert_to_numpy(tensor))

def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    # pdb.set_trace()
    save_image_numpy(ndarr, fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

def run_inference(dataloader, econdata, pipeline, carving, cfg: TestConfig,  save_dir):
    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(cfg.seed)
    
    images_cond, pred_cat = [], defaultdict(list)
    for case_id, batch in tqdm(enumerate(dataloader)):
        images_cond.append(batch['imgs_in'][:, 0]) 
        
        imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (B*Nv, 3, H, W)
        if cfg.with_smpl:
            smpl_in = torch.cat([batch['smpl_imgs_in']]*2, dim=0)
            smpl_in = rearrange(smpl_in, "B Nv C H W -> (B Nv) C H W")
        else:
            smpl_in = None

        normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch['color_prompt_embeddings'] 
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")

        with torch.autocast("cuda"):
            # B*Nv images
            guidance_scale = cfg.validation_guidance_scales
            unet_out = pipeline(
                imgs_in, None, prompt_embeds=prompt_embeddings,
                dino_feature=None, smpl_in=smpl_in,
                generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                **cfg.pipe_validation_kwargs
            )
            
            out = unet_out.images
            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:] 
            if cfg.save_mode == 'concat': ## save concatenated color and normal---------------------
                pred_cat[f"cfg{guidance_scale:.1f}"].append(torch.cat([normals_pred, images_pred], dim=-1)) # b, 3, h, w
                cur_dir = os.path.join(save_dir, f"cropsize-{cfg.validation_dataset.crop_size}-cfg{guidance_scale:.1f}-seed{cfg.seed}-smpl-{cfg.with_smpl}")
                os.makedirs(cur_dir, exist_ok=True)
                for i in range(bsz//num_views):
                    scene =  batch['filename'][i].split('.')[0]

                    img_in_ = images_cond[-1][i].to(out.device)
                    vis_ = [img_in_]
                    for j in range(num_views):
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]
                        
                        vis_.append(color)
                        vis_.append(normal)

                    out_filename = f"{cur_dir}/{scene}.png"
                    vis_ = torch.stack(vis_, dim=0)
                    vis_ = make_grid(vis_, nrow=len(vis_), padding=0, value_range=(0, 1))
                    save_image(vis_, out_filename)
            elif cfg.save_mode == 'rgb':
                for i in range(bsz//num_views):
                    scene =  batch['filename'][i].split('.')[0]

                    img_in_ = images_cond[-1][i].to(out.device)
                    normals, colors = [], []
                    for j in range(num_views):
                        idx = i*num_views + j
                        normal = normals_pred[idx]
                        if j == 0:
                            color = imgs_in[0].to(out.device)
                        else:
                            color = images_pred[idx]
                        if j in [3, 4]:
                            normal = torch.flip(normal, dims=[2])
                            color = torch.flip(color, dims=[2])
                            
                        colors.append(color)
                        if j == 6:
                            normal = F.interpolate(normal.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                        normals.append(normal)
                        
                        ## save color and normal---------------------
                        # normal_filename = f"normals_{view}_masked.png"
                        # rgb_filename = f"color_{view}_masked.png"
                        # save_image(normal, os.path.join(scene_dir, normal_filename))
                        # save_image(color, os.path.join(scene_dir, rgb_filename))
                    normals[0][:, :256, 256:512] =  normals[-1]
                    
                    colors = [remove(convert_to_pil(tensor), session=session) for tensor in colors[:6]]
                    normals = [remove(convert_to_pil(tensor), session=session) for tensor in normals[:6]]
        pose = econdata.__getitem__(case_id)
        carving.optimize_case(scene, pose, colors, normals)
        torch.cuda.empty_cache()   
               
     

def load_pshuman_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda')
    return pipeline

def main(
    cfg: TestConfig
):

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    pipeline = load_pshuman_pipeline(cfg)
    

    if cfg.with_smpl:
        from mvdiffusion.data.testdata_with_smpl import SingleImageDataset
    else:
        from mvdiffusion.data.single_image_dataset import SingleImageDataset
        
    # Get the  dataset
    validation_dataset = SingleImageDataset(
        **cfg.validation_dataset
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    dataset_param = {'image_dir': validation_dataset.root_dir, 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')

    carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)
    run_inference(validation_dataloader, econdata, pipeline, carving, cfg, cfg.save_dir)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)
