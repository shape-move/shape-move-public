import torch
import torch.nn as nn
import sys, argparse, os, imageio, trimesh
import numpy as np
import smplx
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
import pytorch_lightning as pl

from model.engine.mlp_shape_motion_token import Latent_Token_predictor
from model.engine.motion_vae import SCMotionVAE
from model.nnutils.motion_utils import recover_from_ric
from utils.visualization import Renderer, FloorRenderer, demo_draw_to_batch
from utils.io_utils import load_json

def parse_args(args):
    parser = argparse.ArgumentParser()
    args, extras = parser.parse_known_args(args)

    opt = OmegaConf.load(f"configs/inference.yaml")
    cli_conf = OmegaConf.from_cli(extras)
    opt = OmegaConf.merge(opt, cli_conf)
    OmegaConf.resolve(opt)

    if opt.stamp is None:
        setattr(opt, "stamp", f'{datetime.now().strftime("%m%d_%H%M%S")}')
    setattr(opt, "log_dir", f"{opt.output_dir}/{opt.stamp}_{opt.name}")   

    return opt


class Reference_model(nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.renderer = Renderer(
            is_registration=False
        )

        self.token_predictor = Latent_Token_predictor(
            model_type=opt.model.model_type,
            model_path=opt.model.model_path,
            motion_codebook_size=opt.model.motion_codebook_size,
        )

        self.motionvae = SCMotionVAE(nb_code=opt.motion_model.quantizer.nb_code,
            code_dim=opt.motion_model.quantizer.code_dim,
            output_emb_width=opt.motion_model.output_emb_width,
            down_t=opt.motion_model.down_t,
            stride_t=opt.motion_model.stride_t,
            width=opt.motion_model.width,
            depth=opt.motion_model.depth,
            dilation_growth_rate=opt.motion_model.dilation_growth_rate,
            quantizer_type=opt.motion_model.quantizer.type,
            level=opt.motion_model.quantizer.levels,
        )

        self.smpl_model_path = 'data'
        self.smpl_model = smplx.create(model_path=self.smpl_model_path, gender='neutral', num_betas=10,
            model_type='smpl', batch_size=1)

        self.shape_mean = torch.tensor(np.load(os.path.join("data", "Shape_Mean_neutral.npy")), dtype=torch.float32)
        self.shape_std = torch.tensor(np.load(os.path.join("data", "Shape_Std_neutral.npy")), dtype=torch.float32)

        meta_dir = 'pretrain_model/sc_t2m/Comp_v6_KLD01/meta'
        self.motion_mean = torch.tensor(np.load(os.path.join(meta_dir, "mean.npy")), dtype=torch.float32)
        self.motion_std = torch.tensor(np.load(os.path.join(meta_dir, "std.npy")), dtype=torch.float32)


    def render_model(self, beta, idx):        
        body = self.smpl_model(betas=beta[None].to(self.device))
        f = self.smpl_model.faces
        shaped_vertices = body['vertices'].detach()
        pred_mesh = trimesh.Trimesh(shaped_vertices.cpu().numpy()[0], f)
        pred_img = self.renderer.render(pred_mesh)

        img_path = f'{self.opt.log_dir}/{idx:05d}_body_figure.png'
        pred_img.save(img_path)

        return img_path
    
    def render_motion(self, xyz, prompt, idx):
        fname = f'{self.opt.log_dir}/{idx:05d}'
        output_npy_path = fname + '.npy'
        output_mp4_path = fname + '.mp4'
        np.save(output_npy_path, xyz)
        pose_vis = demo_draw_to_batch(xyz, [prompt], [output_mp4_path])


def main(args):
    opt = parse_args(args)
    pl.seed_everything(opt.seed)
    device = torch.device('cuda:0')
    output_dir = opt.log_dir
    os.makedirs(output_dir, exist_ok=True)

    model = Reference_model(opt, device).to(device)
    weight = torch.load(opt.resume_path, map_location='cpu', weights_only=True)
    model.load_state_dict(weight, strict=False)

    prompts = load_json(opt.prompt_path)
    for idx, prompt in prompts.items():
        shape = prompt.get("shape", "")
        motion = prompt.get("motion", "")

        motion_tokens, beta_pred = model.token_predictor.generate_direct(shape, motion, 't2sm')
        beta_pred = beta_pred[0].detach().cpu()
        beta_pred = beta_pred.squeeze() * model.shape_std + model.shape_mean
        np.savez(os.path.join(opt.log_dir, f'{int(idx):05d}_shape.npz'), betas=beta_pred.cpu().numpy(), gender='neutral')

        motion_tokens = torch.clamp(motion_tokens[0], 0, model.opt.motion_model.quantizer.nb_code-1, out=None)
        if len(motion_tokens) > 1:
            motion_ = model.motionvae.decode(motion_tokens, beta_pred.to(motion_tokens.device))
        else:
            motion_ = torch.zeros_like([1,20,263])

        motion_denorm = motion_.detach().clone().cpu()*model.motion_std + model.motion_mean
        motion_xyz = recover_from_ric(motion_denorm, joints_num=22)

        xyz = motion_xyz[:1]
        clip_bs, seq = xyz.shape[:2]
        xyz = xyz.reshape(clip_bs, seq, -1, 3) 
        
        model.render_motion(xyz.cpu().numpy(), motion, int(idx))
        model.render_model(beta_pred, int(idx))


if __name__ == "__main__":
    main(sys.argv[1:])  

