import torch
import numpy as np
import os, trimesh, smplx, argparse, h5py, joblib, re, sys
from tqdm import tqdm
from glob import glob

sys.path.insert(0, f'{os.path.join(os.path.dirname(__file__))}/..')
from utils.smplify_tool import SMPLify3D

def plys2npy(ply_dir, out_path):
    file_list = sorted(glob(f'{ply_dir}/*.ply'), key=lambda x: int(re.sub("[^0-9]", "", os.path.basename(x))))

    meshs = np.zeros((len(file_list), 6890, 3))
    for i, path in enumerate(file_list):
        mesh = trimesh.load_mesh(path, process=False)
        vs = mesh.vertices
        assert vs.shape == (6890, 3)
        meshs[i] = vs 

    file_name = os.path.join(out_path.replace('.npy', "_mesh.npy"))
    np.save(file_name, meshs)

def motion_fit_mesh(paths, save_paths, batch_size, num_smplify_iters):
    """ fit motion data to mesh path
    Args:
        paths: [List] path to motion data with shape [nframes, 22, 3]
        save_paths: [List] path to save folder, the generated mesh will be saved under ./folder/meshes
    """
    num_joints = 22
    pbar = tqdm(paths)
    init_pose = np.load('data/smpl/pose.npy')
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)
    init_mean_pose = (torch.from_numpy(init_pose).unsqueeze(0).float().repeat(batch_size, 1).to(device))

    for n, p in enumerate(pbar):
        pbar.set_description(f"processing path {p}")
        if os.path.isfile(p.replace('.npy', '_mesh.npy')):
            print("already generated mesh")
            continue

        motion = np.load(p)
        if len(motion.shape) > 3:
            motion = motion[0]

        optimize_shape = False
        bdata = np.load(p.replace('.npy', '_shape.npz'), allow_pickle=True)

        gender = 'neutral'
        beta = bdata['betas'][:10]
        beta = torch.Tensor(beta).to(device).float()
        if len(beta.shape) < 2:
            beta = beta[None]

        smplmodel = smplx.create('data', model_type="smpl",
            gender=gender, ext="pkl", batch_size=1,
        ).to(device)
        smplify = SMPLify3D(
            smplxmodel=smplmodel,
            batch_size=batch_size,
            joints_category="AMASS",
            num_iters=num_smplify_iters,
            device=device,
        )

        pred_pose = torch.zeros(batch_size, 72).to(device)
        # pred_betas = torch.zeros(batch_size, 10).to(device)
        pred_cam_t = torch.zeros(batch_size, 3).to(device)
        keypoints_3d = torch.zeros(batch_size, num_joints, 3).to(device)

        for idx in tqdm(range(motion.shape[0])):
            joints3d = motion[idx] 
            keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

            if idx == 0:
                pred_pose[0, :] = init_mean_pose
                pred_cam_t[0, :] = cam_trans_zero
            else:
                # use previous result
                data_param = joblib.load(save_paths[n]+f"/smplfit/motion_{(idx - 1):04d}"+ ".pkl")
                pred_pose[0, :] = torch.from_numpy(
                                data_param["pose"]).unsqueeze(0).float()
                pred_cam_t[0, :] = torch.from_numpy(
                    data_param["cam"]).unsqueeze(0).float()
                
                if optimize_shape:
                    beta[0, :] = torch.from_numpy(
                    data_param["beta"]).unsqueeze(0).float()

            confidence_input = torch.ones(num_joints)

            (new_opt_vertices, new_opt_joints,
                new_opt_pose, new_opt_betas,
                new_opt_cam_t, new_opt_joint_loss,
            ) = smplify(
                pred_pose.detach(), beta.detach(),
                pred_cam_t.detach(),
                keypoints_3d, conf_3d=confidence_input.to(device),
                seq_ind=idx if optimize_shape else 1,
                # seq_ind=1, # to fix the shape
            )
            
            output_model = smplmodel(
                betas=new_opt_betas,
                global_orient=new_opt_pose[:, :3],
                body_pose=new_opt_pose[:, 3:],
                transl=new_opt_cam_t[0],
                return_verts=True,
            )
            mesh_p = trimesh.Trimesh(
                vertices=output_model.vertices.detach().cpu().numpy().squeeze(),
                faces=smplmodel.faces,
                process=False,
            )
            ply_path = os.path.join(save_paths[n], 'smplfit', f"motion_{idx:04d}.ply")
            os.makedirs(os.path.dirname(ply_path), exist_ok=True)
            mesh_p.export(ply_path)
            param = {}
            param["beta"] = new_opt_betas.detach().cpu().numpy()
            param["pose"] = new_opt_pose.detach().cpu().numpy()
            param["cam"] = new_opt_cam_t.detach().cpu().numpy()
            joblib.dump(param, ply_path.replace('ply', 'pkl'), compress=3)

        # merge plys to one npy
        plys2npy(os.path.join(save_paths[n],'smplfit'), p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_smplify_iters', type=int, default=100, help='num of iter to fit mesh')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num', nargs='+', default=[0])
    parser.add_argument('--dir', type=str)
    
    arg = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    paths = []
    save_paths = []

    for f in os.listdir(arg.dir):
        if f.endswith('.npy') and not f.endswith('_mesh.npy'):
            paths.append(f'{arg.dir}/{f}')
            save_paths.append(f'{arg.dir}/{f.split(".")[0]}_result')

    motion_fit_mesh(paths, save_paths, arg.batch_size, arg.num_smplify_iters)


    