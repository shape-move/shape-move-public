import torch
import torch.nn as nn
import os, sys
import pickle, smplx, h5py
import numpy as np
from tqdm import tqdm
from model.nnutils.quaternion import axis_angle_to_matrix, matrix_to_rotation_6d

AMASS_JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1,
    'LKnee': 4,
    'LAnkle': 7,
    'LFoot': 10,
    'RHip': 2,
    'RKnee': 5,
    'RAnkle': 8,
    'RFoot': 11,
    'LShoulder': 16,
    'LElbow': 18,
    'LWrist': 20,
    'RShoulder': 17,
    'RElbow': 19,
    'RWrist': 21,
    'spine1': 3,
    'spine2': 6,
    'spine3': 9,
    'Neck': 12,
    'Head': 15,
    'LCollar': 13,
    'Rcollar': 14,
}

# Map joints Name to SMPL joints idx
JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1,
    'LKnee': 4,
    'LAnkle': 7,
    'LFoot': 10,
    'RHip': 2,
    'RKnee': 5,
    'RAnkle': 8,
    'RFoot': 11,
    'LShoulder': 16,
    'LElbow': 18,
    'LWrist': 20,
    'LHand': 22,
    'RShoulder': 17,
    'RElbow': 19,
    'RWrist': 21,
    'RHand': 23,
    'spine1': 3,
    'spine2': 6,
    'spine3': 9,
    'Neck': 12,
    'Head': 15,
    'LCollar': 13,
    'Rcollar': 14,
    'Nose': 24,
    'REye': 26,
    'LEye': 26,
    'REar': 27,
    'LEar': 28,
    'LHeel': 31,
    'RHeel': 34,
    'OP RShoulder': 17,
    'OP LShoulder': 16,
    'OP RHip': 2,
    'OP LHip': 1,
    'OP Neck': 12,
}

# SMPLIfy 3D
class SMPLify3D():
    """Implementation of SMPLify, use 3D joints."""

    def __init__(self,
                 smplxmodel,
                 step_size=1e-2,
                 batch_size=1,
                 num_iters=100,
                 use_collision=False,
                 use_lbfgs=True,
                 joints_category="orig",
                 device=torch.device('cuda:0'),
                 ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        prior_folder='data/smpl'
        self.pose_prior = MaxMixturePrior(prior_folder=prior_folder,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # collision part
        self.use_collision = use_collision
        if self.use_collision:
            self.part_segm_fn = config.Part_Seg_DIR
        
        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category
        
        if joints_category=="orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx 
        elif joints_category=="AMASS":
            self.smpl_index = range(22)
            self.corr_index = range(22)
        else:
            self.smpl_index = None 
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0, seq_ind=0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
			seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        # # # add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None
        
        if self.use_collision:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=8)

            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                           sigma=0.5, point2plane=False, vectorized=True, penalize_outside=True)

            if self.part_segm_fn:
                # Read the part segmentation
                part_segm_fn = os.path.expandvars(self.part_segm_fn)
                with open(part_segm_fn, 'rb') as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file,  encoding='latin1')
                faces_segm = face_segm_data['segm']
                faces_parents = face_segm_data['parents']
                # Create the module used to filter invalid collision pairs
                filter_faces = FilterFaces(
                    faces_segm=faces_segm, faces_parents=faces_parents,
                    ign_part_pairs=None).to(device=self.device)
                    
                    
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()
        
        preserve_pose = init_pose[:, 3:].detach().clone()
       # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.num_iters,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(10):
                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints

                    loss = camera_fitting_loss_3d(model_joints, camera_translation,
                                                  init_cam_t, j3d, self.joints_category)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(20):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(model_joints[:, self.smpl_index], camera_translation,
                                              init_cam_t,  j3d[:, self.corr_index], self.joints_category)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # --- if we use the sequence, fix the shape
        if seq_ind == 0:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]
        # print(betas)
        # print("seq_ind: ", seq_ind)

        

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.num_iters,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            
            for i in (range(self.num_iters)):
                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                                j3d[:, self.corr_index], self.pose_prior,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                pose_preserve_weight=5.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                            j3d[:, self.corr_index], self.pose_prior,
                                            joints3d_conf=conf_3d,
                                            joint_loss_weight=600.0,
                                            use_collision=self.use_collision, 
                                            model_vertices=model_vertices, model_faces=self.model_faces,
                                            search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                              j3d[:, self.corr_index], self.pose_prior,
                                              joints3d_conf=conf_3d,
                                              joint_loss_weight=600.0,
                                              use_collision=self.use_collision, model_vertices=model_vertices, model_faces=self.model_faces,
                                              search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss

# #####--- get camera fitting loss -----
def camera_fitting_loss_3d(model_joints, camera_t, camera_t_est,
                           j3d, joints_category="orig", depth_loss_weight=100.0):
    """
    Loss function for camera optimization.
    """
    model_joints = model_joints + camera_t
    # # get the indexed four
    # op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    # op_joints_ind = [config.JOINT_MAP[joint] for joint in op_joints]
    #
    # j3d_error_loss = (j3d[:, op_joints_ind] -
    #                          model_joints[:, op_joints_ind]) ** 2

    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="orig":
        select_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        select_joints_ind = [AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    j3d_error_loss = (j3d[:, select_joints_ind] -
                      model_joints[:, gt_joints_ind]) ** 2

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight**2) *  (camera_t - camera_t_est)**2

    total_loss = j3d_error_loss +  depth_loss
    return total_loss.sum()

@torch.no_grad()
def guess_init_3d(model_joints, 
                  j3d, 
                  joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="orig":
        joints_ind_category = [JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        joints_ind_category = [AMASS_JOINT_MAP[joint] for joint in gt_joints] 
    else:
        print("NO SUCH JOINTS CATEGORY!") 

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t

DEFAULT_DTYPE = torch.float32
class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)

# Guassian
def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
    
# angle prior
def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


# #####--- body fitiing loss -----
def body_fitting_loss_3d(body_pose, preserve_pose,
                         betas, model_joints, camera_translation,
                         j3d, pose_prior,
                         joints3d_conf,
                         sigma=100, pose_prior_weight=4.78*1.5,
                         shape_prior_weight=5.0, angle_prior_weight=15.2,
                         joint_loss_weight=500.0,
                         pose_preserve_weight=0.0,
                         use_collision=False,
                         model_vertices=None, model_faces=None,
                         search_tree=None,  pen_distance=None,  filter_faces=None,
                         collision_loss_weight=1000
                         ):
    """
    Loss function for body fitting
    """
    batch_size = body_pose.shape[0]

    #joint3d_loss = (joint_loss_weight ** 2) * gmof((model_joints + camera_translation) - j3d, sigma).sum(dim=-1)
    
    joint3d_error = gmof((model_joints + camera_translation) - j3d, sigma)
    
    joint3d_loss_part = (joints3d_conf ** 2) * joint3d_error.sum(dim=-1)
    # joint3d_loss = (joint_loss_weight ** 2) * joint3d_loss_part
    joint3d_loss = ((joint_loss_weight ** 2) * joint3d_loss_part).sum(dim=-1)
    
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    collision_loss = 0.0
    # Calculate the loss due to interpenetration
    if use_collision:
        triangles = torch.index_select(
            model_vertices, 1,
            model_faces).view(batch_size, -1, 3, 3)

        with torch.no_grad():
            collision_idxs = search_tree(triangles)

        # Remove unwanted collisions
        if filter_faces is not None:
            collision_idxs = filter_faces(collision_idxs)

        if collision_idxs.ge(0).sum().item() > 0:
            collision_loss = torch.sum(collision_loss_weight * pen_distance(triangles, collision_idxs))
    
    pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

    total_loss = joint3d_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + collision_loss + pose_preserve_loss

    return total_loss.sum()

class joints2smpl:

    def __init__(self, num_frames, device, smpl_model_path, smpl_mean_file, gender=None):
        self.device = device
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 100
        self.fix_foot = False
        gender = gender if gender is not None else 'neutral'
        self.smplmodel = smplx.create(smpl_model_path,
                                 model_type="smpl", gender=gender, ext="pkl",
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=self.smplmodel,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)


    def joint2smpl(self, input_joints, betas=None, init_params=None):
        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(self.batch_size, self.num_joints, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]


        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        betas = pred_betas if betas is None else betas.repeat(self.batch_size, 1).float().to(self.device)
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            seq_ind=1
        )

        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        thetas = matrix_to_rotation_6d(axis_angle_to_matrix(thetas))  # [bs, 24, 6]
        root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [bs, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, 196]

        # output_model = self.smplmodel(
        #     betas=new_opt_betas,
        #     global_orient=new_opt_pose[..., :3],
        #     body_pose=new_opt_pose[..., 3:],
        #     transl=new_opt_cam_t,
        #     return_verts=True,
        # )

        # vertices = output_model.vertices.detach().cpu().numpy()
        # # .squeeze()

        return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}

SMPL_BODY_BONES = [-0.0018, -0.2233, 0.0282, 0.0695, -0.0914, -0.0068, -0.0677, -0.0905, -0.0043,
                   -0.0025, 0.1090, -0.0267, 0.0343, -0.3752, -0.0045, -0.0383, -0.3826, -0.0089,
                   0.0055, 0.1352, 0.0011, -0.0136, -0.3980, -0.0437, 0.0158, -0.3984, -0.0423,
                   0.0015, 0.0529, 0.0254, 0.0264, -0.0558, 0.1193, -0.0254, -0.0481, 0.1233,
                   -0.0028, 0.2139, -0.0429, 0.0788, 0.1217, -0.0341, -0.0818, 0.1188, -0.0386,
                   0.0052, 0.0650, 0.0513, 0.0910, 0.0305, -0.0089, -0.0960, 0.0326, -0.0091,
                   0.2596, -0.0128, -0.0275, -0.2537, -0.0133, -0.0214, 0.2492, 0.0090, -0.0012,
                   -0.2553, 0.0078, -0.0056, 0.0840, -0.0082, -0.0149, -0.0846, -0.0061, -0.0103]


class HybrIKJointsToRotmat:
    def __init__(self):
        self.naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.num_nodes = 22
        self.parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15,
                      16, 17, -2, 18, 19, 20, 21, -2, -2]
        self.bones = np.reshape(np.array(SMPL_BODY_BONES), [24, 3])[:self.num_nodes]

    def multi_child_rot(self, t, p,
                        pose_global_parent):
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = np.matmul(t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1]))
        u, s, vt = np.linalg.svd(m)
        r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))
        err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = np.reshape(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                            [1, 3, 3])
        r_fix = np.matmul(np.transpose(vt, [0, 2, 1]),
                          np.matmul(id_fix,
                                    np.transpose(u, [0, 2, 1])))
        r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
        return r, np.matmul(pose_global_parent, r)

    def single_child_rot(self, t, p, pose_global_parent, twist=None):
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
        sina = np.linalg.norm(cross, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                               np.linalg.norm(p_rot, axis=1, keepdims=True))
        cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                           np.linalg.norm(p_rot, axis=1, keepdims=True))
        sina = np.reshape(sina, [-1, 1, 1])
        cosa = np.reshape(cosa, [-1, 1, 1])
        skew_sym_t = np.stack([0.0 * cross[:, 0], -cross[:, 2], cross[:, 1],
                               cross[:, 2], 0.0 * cross[:, 0], -cross[:, 0],
                               -cross[:, 1], cross[:, 0], 0.0 * cross[:, 0]], 1)
        skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
        dsw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                 skew_sym_t)
        if twist is not None:
            skew_sym_t = np.stack([0.0 * t[:, 0], -t[:, 2], t[:, 1],
                                   t[:, 2], 0.0 * t[:, 0], -t[:, 0],
                                   -t[:, 1], t[:, 0], 0.0 * t[:, 0]], 1)
            skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
            sina = np.reshape(twist[:, 1], [-1, 1, 1])
            cosa = np.reshape(twist[:, 0], [-1, 1, 1])
            dtw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                    ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                     skew_sym_t)
            dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
        return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)

    def __call__(self, joints, twist=None):
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = np.expand_dims(joints, 0)
            if twist is not None:
                twist = np.expand_dims(twist, 0)
        assert (len(joints.shape) == 3)
        batch_size = np.shape(joints)[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = 0.0 * joints_rel
        pose_global = np.zeros([batch_size, self.num_nodes, 3, 3])
        pose = np.zeros([batch_size, self.num_nodes, 3, 3])
        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = np.matmul(pose_global[:, self.parents[i]],
                                                np.reshape(self.bones[i], [1, 3, 1])).reshape(-1, 3) + \
                                      joints_hybrik[:, self.parents[i]]
            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue
            if i == 0:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                                             np.eye(3).reshape(1, 3, 3))

            elif i == 9:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                                             pose_global[:, self.parents[9]])
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                r, rg = self.single_child_rot(self.bones[self.child[i]].reshape(1, 3, 1),
                                              p.reshape(-1, 3, 1),
                                              pose_global[:, self.parents[i]],
                                              twi)
            pose[:, i] = r
            pose_global[:, i] = rg
        if expand_dim:
            pose = pose[0]
        return pose
