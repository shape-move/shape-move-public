from typing import NewType, Dict, Tuple
import os.path as osp
import yaml, contextlib, os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from smplx import SMPLLayer as _SMPLLayer
from smplx.lbs import vertices2joints

from model.nnutils.quaternion import axis_angle_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix

Tensor = NewType('Tensor', torch.Tensor)

Attribute_map = {
        'chest': 'chest',
        'waist': 'waist',
        'hip': 'hips',
        'height': 'height',
        'arm': 'arm right length',
        'leg': 'inside leg height'
}

# define the location and vertice id
SMPL_LANDMARK_INDICES = {"HEAD_TOP": 412,
                    "HEAD_LEFT_TEMPLE": 166,
                    "NECK_ADAM_APPLE": 3050,
                    "LEFT_HEEL": 3458,
                    "RIGHT_HEEL": 6858,
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1103,
                    "LEFT_ANKLE": 3325
                    }

SMPL_LANDMARK_INDICES["HEELS"] = (SMPL_LANDMARK_INDICES["LEFT_HEEL"], 
                                  SMPL_LANDMARK_INDICES["RIGHT_HEEL"])
LENGTHS = {"height": 
                    (SMPL_LANDMARK_INDICES["HEAD_TOP"], 
                     SMPL_LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height": 
                    (SMPL_LANDMARK_INDICES["SHOULDER_TOP"], 
                     SMPL_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length": 
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm":
                    (SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "leg": 
                    (SMPL_LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     SMPL_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

class BodyMeasurements(nn.Module):
    """ Adapted from https://github.com/muelea/shapy/blob/master/mesh-mesh-intersection/body_measurements/body_measurements.py
    compute smpl body length and circumference
    """
    # The density of the human body is 985 kg / m^3
    DENSITY = 985

    def __init__(self, cfg, **kwargs):
        ''' Loss that penalizes deviations in weight and height
        '''
        super(BodyMeasurements, self).__init__()
        from mesh_mesh_intersection import MeshMeshIntersection


        meas_definition_path = cfg.get('meas_definition_path', '')
        meas_definition_path = osp.expanduser(
            osp.expandvars(meas_definition_path))
        meas_vertices_path = cfg.get('meas_vertices_path', '')
        meas_vertices_path = osp.expanduser(
            osp.expandvars(meas_vertices_path))

        with open(meas_definition_path, 'r') as f:
            measurements_definitions = yaml.safe_load(f, )

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.safe_load(f)

        head_top = meas_vertices['HeadTop']
        left_heel = meas_vertices['HeelLeft']

        left_heel_bc = left_heel['bc']
        self.left_heel_face_idx = left_heel['face_idx']

        left_heel_bc = torch.tensor(left_heel['bc'], dtype=torch.float32)
        self.register_buffer('left_heel_bc', left_heel_bc)

        head_top_bc = torch.tensor(head_top['bc'], dtype=torch.float32)
        self.register_buffer('head_top_bc', head_top_bc)

        self.head_top_face_idx = head_top['face_idx']

        action = measurements_definitions['CW_p']
        chest_periphery_data = meas_vertices[action[0]]

        self.chest_face_index = chest_periphery_data['face_idx']
        chest_bcs = torch.tensor(
            chest_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('chest_bcs', chest_bcs)

        action = measurements_definitions['BW_p']
        belly_periphery_data = meas_vertices[action[0]]

        self.belly_face_index = belly_periphery_data['face_idx']
        belly_bcs = torch.tensor(
            belly_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('belly_bcs', belly_bcs)

        action = measurements_definitions['IW_p']
        hips_periphery_data = meas_vertices[action[0]]

        self.hips_face_index = hips_periphery_data['face_idx']
        hips_bcs = torch.tensor(
            hips_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('hips_bcs', hips_bcs)

        max_collisions = cfg.get('max_collisions', 256)
        self.isect_module = MeshMeshIntersection(max_collisions=max_collisions)

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Human Body Density: {self.DENSITY}')
        return '\n'.join(msg)

    def _get_plane_at_heights(self, height: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        device = height.device
        batch_size = height.shape[0]

        verts = torch.tensor(
            [[-1., 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
            device=device).unsqueeze(dim=0).expand(batch_size, -1, -1).clone()
        verts[:, :, 1] = height.reshape(batch_size, -1)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device,
                             dtype=torch.long)

        return verts, faces, verts[:, faces]

    def compute_peripheries(
        self,
        triangles: Tensor,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
    ) -> Dict[str, Tensor]:
        '''
            Parameters
            ----------
                triangles: BxFx3x3 torch.Tensor
                Contains the triangle coordinates for a batch of meshes with
                the same topology
        '''

        batch_size, num_triangles = triangles.shape[:2]
        device = triangles.device

        batch_indices = torch.arange(
            batch_size, dtype=torch.long,
            device=device).reshape(-1, 1) * num_triangles

        meas_data = {}
        if compute_chest:
            meas_data['chest'] = (self.chest_face_index, self.chest_bcs)
        if compute_waist:
            meas_data['waist'] = (self.belly_face_index, self.belly_bcs)
        if compute_hips:
            meas_data['hip'] = (self.hips_face_index, self.hips_bcs)

        output = {}
        for name, (face_index, bcs) in meas_data.items():

            vertex = (
                triangles[:, face_index] * bcs.reshape(1, 3, 1)).sum(axis=1)

            _, _, plane_tris = self._get_plane_at_heights(vertex[:, 1])

            with torch.no_grad():
                collision_faces, collision_bcs = self.isect_module(
                    plane_tris, triangles)

            selected_triangles = triangles.view(-1, 3, 3)[
                (collision_faces + batch_indices).view(-1)].reshape(
                    batch_size, -1, 3, 3)
            points = (
                selected_triangles[:, :, None] *
                collision_bcs[:, :, :, :, None]).sum(
                axis=-2).reshape(batch_size, -1, 2, 3)

            np_points = points.detach().cpu().numpy()
            collision_faces = collision_faces.detach().cpu().numpy()
            collision_bcs = collision_bcs.detach().cpu().numpy()

            output[name] = {
                'points': [],
                'valid_points': [],
                'value': [],
                'plane_height': vertex[:, 1],
            }

            # with torch.no_grad():
            for ii in range(batch_size):
                valid_face_idxs = np.where(collision_faces[ii] > 0)[0]
                points_in_plane = np_points[
                    ii, valid_face_idxs, :, ][:, :, [0, 2]].reshape(
                        -1, 2)
                hull = ConvexHull(points_in_plane)
                point_indices = hull.simplices.reshape(-1)

                hull_points = points[ii][valid_face_idxs].view(
                    -1, 3)[point_indices].reshape(-1, 2, 3)

                meas_value = (
                    hull_points[:, 1] - hull_points[:, 0]).pow(2).sum(
                    dim=-1).sqrt().sum()

                # output[name]['valid_points'].append(
                #     np_points[ii, valid_face_idxs])
                # output[name]['points'].append(hull_points)
                output[name]['value'].append(meas_value)
            # *100 to turn into cm
            output[name]['tensor'] = torch.stack(output[name]['value'])*100
        return output

    def compute_height(self, shaped_triangles: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Compute the height using the heel and the top of the head
        '''
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (head_top_tri[:, 0, :] * self.head_top_bc[0] +
                    head_top_tri[:, 1, :] * self.head_top_bc[1] +
                    head_top_tri[:, 2, :] * self.head_top_bc[2])
        head_top = (
            head_top_tri * self.head_top_bc.reshape(1, 3, 1)
        ).sum(dim=1)
        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (
            left_heel_tri * self.left_heel_bc.reshape(1, 3, 1)
        ).sum(dim=1)

        return (torch.abs(head_top[:, 1] - left_heel[:, 1])*100,
                torch.stack([head_top, left_heel], axis=0)
                )

    def compute_mass(self, tris: Tensor) -> Tensor:
        ''' Computes the mass from volume and average body density
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def measure_length(self, vertices, compute_arms, compute_legs):
        '''
        Adapted from https://github.com/DavidBoja/SMPL-Anthropometry/tree/449d88f27522f5862433a9dbaacb3d983079a0c5
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''
        measurement_landmarks_inds = {}
        length = {}
        if compute_arms:
            measurement_landmarks_inds['arm'] = LENGTHS['arm']
        if compute_legs:
            measurement_landmarks_inds['leg'] = LENGTHS['leg']
            
        for name, landmarks_ind in measurement_landmarks_inds.items():
            a = vertices[:, landmarks_ind[0], :]
            b = vertices[:, landmarks_ind[1], :]

            dist = torch.norm(b-a, dim=1)
            # dummy code to match the format in measurement
            length[name] = {}
            length[name]['tensor'] = dist*100

        return length

    def forward(
        self,
        triangles: Tensor,
        vertices: Tensor,
        compute_mass: bool = True,
        compute_height: bool = True,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
        compute_arms: bool = True,
        compute_legs: bool = True,
        **kwargs
    ):
        measurements = {}
        if compute_mass:
            measurements['mass'] = {}
            mesh_mass = self.compute_mass(triangles)
            measurements['mass']['tensor'] = mesh_mass

        if compute_height:
            measurements['height'] = {}
            mesh_height, points = self.compute_height(triangles)
            measurements['height']['tensor'] = mesh_height
            measurements['height']['points'] = points

        output = self.compute_peripheries(triangles,
                                          compute_chest=compute_chest,
                                          compute_waist=compute_waist,
                                          compute_hips=compute_hips,
                                          )
        measurements.update(output)
        length = self.measure_length(vertices, compute_arms=compute_arms, compute_legs=compute_legs)
        measurements.update(length)

        return {'measurements': measurements}

action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]


JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
class SMPL(_SMPLLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, model_path, J_regressor_extra_path, **kwargs):
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)
            
        J_regressor_extra = np.load(J_regressor_extra_path)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
        a2m_indexes = vibe_indexes[action2motion_joints]
        smpl_indexes = np.arange(24)
        a2mpl_indexes = np.unique(np.r_[smpl_indexes, a2m_indexes])

        self.maps = {"vibe": vibe_indexes,
                     "a2m": a2m_indexes,
                     "smpl": smpl_indexes,
                     "a2mpl": a2mpl_indexes}
        
    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {"vertices": smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]
            
        return output
    
JOINTSTYPE_ROOT = {"a2m": 0, # action2motion
                   "smpl": 0,
                   "a2mpl": 0, # set(smpl, a2m)
                   "vibe": 8}  # 0 is the 8 position: OP MidHip below
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

class Rotation2xyz:
    def __init__(self, device, model_path, J_regressor_extra_path, dataset='amass'):
        self.device = device
        self.dataset = dataset
        model_path = os.path.join(model_path, 'smpl')
        self.smpl_model = SMPL(model_path, J_regressor_extra_path).eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
        else:
            betas = betas.repeat(rotations.shape[0], 1)
            
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz