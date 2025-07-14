import torch 
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio
import os, json, natsort, shutil, re
import codecs as cs
import trimesh, smplx
from tqdm import tqdm
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
from scipy.spatial.transform.rotation import Rotation as RRR
from shapely import geometry

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
from pyrender import OffscreenRenderer
import PIL.Image as pil_img
from PIL import Image

class Renderer():
    def __init__(
        self,
        is_registration=False,
        rotation=0,
    ):

        self.is_registration = is_registration
        self.rotation = rotation

        H, W = 1200,1200

        self.material = pyrender.MetallicRoughnessMaterial(
                        doubleSided=True,
                        metallicFactor=0.02,
                        roughnessFactor=0.7,
                        smooth=False,
                        alphaMode='BLEND',
                        baseColorFactor=(0.4, 0.4, 0.4, 1.0), #grey color
                        #baseColorFactor=(1.0, 1.0, 1.0, 1.0), #white color
                        # baseColorFactor=(0.45, 0.5, 1.0, 1.0), #blue color
        )
        self.vertex_material = pyrender.MetallicRoughnessMaterial(
            doubleSided=True,
            metallicFactor=0.9,
            roughnessFactor=0.7,
            smooth=False,
            alphaMode='BLEND',
            #baseColorFactor=(1.0, 1.0, 1.0, 1.0)) #white color
            #baseColorFactor=(0.45, 0.5, 1.0, 1.0)) #blue color
        )
        self.material_plane = pyrender.MetallicRoughnessMaterial(
                        doubleSided=True,
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        self.scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                               ambient_light=(0.3, 0.3, 0.3))

        # create camera
        camera = pyrender.PerspectiveCamera(yfov= 0.8 * np.pi / 3.0, aspectRatio=1.0)

        # set camera pose
        camera_pose = np.array(
            [[1, 0, 0, 0],
            [ 0, 1, 0, 1.2],
            [ 0, 0, 1, 2.5],
            [ 0, 0, 0, 1]],
        )
        camera_rot = trimesh.transformations.rotation_matrix(
                      np.radians(355), [1, 0, 0])
        camera_pose[:3,:3] = camera_rot[:3,:3]
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(nc)

        # 'sun light'
        light = pyrender.light.DirectionalLight(intensity=3)
        sl_pose = np.array(
            [[1, 0, 0, 0],
            [ 0, 1, 0, 1.5],
            [ 0, 0, 1, 6],
            [ 0, 0, 0, 1]],
        )
        sl_rot = trimesh.transformations.rotation_matrix(
             np.radians(340), [1, 0, 0])
        sl_pose[:3,:3] = sl_rot[:3,:3]
        nl = pyrender.Node(light=light, matrix=sl_pose)
        self.scene.add_node(nl)

        xs = 0.5
        sl2_pose = np.eye(4)
        sl2_poses = {
            'pointlight': [[-1.0, 1.0, 1.0], [-1.0, 2.0, 1.0]]
        }

        # add ground plane
        plane_vertices = np.zeros([4, 3], dtype=np.float32)
        ps = 1
        plane_vertices[:, 0] = [-ps, ps, ps, -ps]
        plane_vertices[:, 2] = [-ps, -ps, ps, ps]
        plane_faces = np.array([[0, 1, 2], [0, 2, 3]],
                                   dtype=np.int32).reshape(-1, 3)
        plane_mesh = trimesh.Trimesh(vertices=plane_vertices,
                                         faces=plane_faces)
        pyr_plane_mesh = pyrender.Mesh.from_trimesh(
                        plane_mesh,
                        material=self.material_plane)
        npl = pyrender.Node(mesh=pyr_plane_mesh, matrix=np.eye(4))

        # create renderer
        self.r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)

    def render(self, mesh, rotation=0, colors=None, vertex_colors=None):

        if self.rotation != 0:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rotation), [0, 1, 0])
            mesh.apply_transform(rot)

        if self.is_registration:
            # rotate 90 deg around x axis
            rotaround = 270
            rot = trimesh.transformations.rotation_matrix(
                  np.radians(rotaround), [1, 0, 0])
            mesh.apply_transform(rot)

        # set feet to zero
        vertices = np.array(mesh.vertices)
        vertices[:, 1] = vertices[:, 1] - vertices[:, 1].min()
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                               process=False,
                               vertex_colors=vertex_colors,
                               )
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors

        # create pyrender mesh
        pyr_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            #  material=self.material if vertex_colors is None else None,
            material=self.material if vertex_colors is None else 
            self.vertex_material,
            smooth=True)

        # remove old mesh if still in scene
        try:
            self.scene.remove_node(self.nm)
        except:
            pass
        self.nm = pyrender.Node(mesh=pyr_mesh, matrix=np.eye(4))
        self.scene.add_node(self.nm)

        # render and save
        color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        output_img = pil_img.fromarray((color).astype(np.uint8))
        w, h = output_img.size
        output_img.crop((200,0,w-200,h)) # \
            #.save(osp.join(self.output_folder, m.replace('ply', 'png')))

        return output_img

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            # ax.set_xlim(-limits, limits)
            # ax.set_ylim(-limits, limits)
            # ax.set_zlim(0, limits)
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        if title is not None :
            wraped_title = title
            # wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16, horizontalalignment='left', x=0.05)
        ax = p3.Axes3D(fig)
        fig.add_axes(ax)
        
        init()
        
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # plt.savefig('test.png', dpi=96)
        # quit()
        if out_name is not None : 
            plt.savefig(out_name, dpi=96)
            plt.close()
            
        else : 
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


class FloorRenderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self,
                 focal_length=5000,
                 img_res=(224, 224),
                 faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                                   viewport_height=img_res[1],
                                                   point_size=2.0)
        
        print("create renderer")

        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces

        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

    def init_floor(self, vertices):
        MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
        MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]

        out_list = []
        minx = MINS[0] - 0.5
        maxx = MAXS[0] + 0.5
        minz = MINS[2] - 0.5 
        maxz = MAXS[2] + 0.5
        miny = MINS[1].cpu().numpy()

        floor = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz],
                                  [maxx, minz]])
        self.floor = trimesh.creation.extrude_polygon(floor, 1e-5)
        self.floor.visual.face_colors = [0, 0, 0, 0.2]
        # self.floor.apply_transform(self.rot)
        self.floor_pose = np.array(
            [[1, 0, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), miny],
             [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]])

        c = -np.pi / 6

        self.camera_pose = [[1, 0, 0, (minx + maxx).cpu().numpy() / 2],
                            [0, np.cos(c), -np.sin(c), 1.5],
                            [
                                0,
                                np.sin(c),
                                np.cos(c),
                                max(4, minz.cpu().numpy() + (1.5 - miny) * 2, (maxx - minx).cpu().numpy())
                            ], [0, 0, 0, 1]]

    def __call__(self, vertices, camera_translation=None):
        floor_render = pyrender.Mesh.from_trimesh(self.floor, smooth=False)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.02,
            roughnessFactor=0.7,
            alphaMode='BLEND',
            baseColorFactor=(0.122, 0.329, 0.539, 1.0))
        mesh = trimesh.Trimesh(vertices, self.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=200)
        spot_l = pyrender.SpotLight(color=np.ones(3),
                                    intensity=300.0,
                                    innerConeAngle=np.pi / 16,
                                    outerConeAngle=np.pi / 6)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=300.0)

        scene = pyrender.Scene(bg_color=(1., 1., 1., 0.8),
                               ambient_light=(0.4, 0.4, 0.4))
        scene.add(floor_render, pose=self.floor_pose)
        scene.add(mesh, 'mesh')

        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        scene.add(camera, pose=self.camera_pose)

        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        color, rend_depth = self.renderer.render(scene, flags=flags)

        return color


class SMPLRender():

    def __init__(self, SMPL_MODEL_DIR, gender=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        gender = gender if gender is not None else 'neutral'
        self.smpl = smplx.create(SMPL_MODEL_DIR,
                                 model_type="smpl",
                                 gender=gender,
                                 batch_size=1).to(self.device)

        self.pred_camera_t = []
        self.focal_length = 110

    def init_renderer(self, res, smpl_param, betas=None, is_headroot=False):
        poses = smpl_param['pred_pose']
        pred_rotmats = []
        for pose in poses:
            if pose.size == 72:
                pose = pose.reshape(-1, 3)
                pose = RRR.from_rotvec(pose).as_matrix()
                pose = pose.reshape(1, 24, 3, 3)
            pred_rotmats.append(
                torch.from_numpy(pose.astype(np.float32)[None]).to(
                    self.device))
        pred_rotmat = torch.cat(pred_rotmats, dim=0)

        pred_betas = torch.from_numpy(smpl_param['pred_shape'].reshape(
            1, 10).astype(np.float32)).to(self.device)
        pred_root = torch.tensor(smpl_param['pred_root'].reshape(-1, 3).astype(
            np.float32),
                             device=self.device)

        betas = betas if betas is not None else pred_betas

        smpl_output = self.smpl(betas=betas,
                                body_pose=pred_rotmat[:, 1:],
                                transl=pred_root,
                                global_orient=pred_rotmat[:, :1],
                                pose2rot=False)
        
        self.vertices = smpl_output.vertices.detach().cpu().numpy()

        pred_root = pred_root[0]

        if is_headroot:
            pred_root = pred_root - smpl_output.joints[
                0, 12].detach().cpu().numpy()

        self.pred_camera_t.append(pred_root)
        
        print("finish init")

        return self.vertices

    def render(self, index):
        renderImg = self.renderer(self.vertices[index, ...],
                                  self.pred_camera_t)
        return renderImg

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None) : 
    # print(smpl_joints_batch.shape)
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch]))
        if outname is not None:
            writer = imageio.get_writer(outname, fps=20)
            for f in out[-1]:
                writer.append_data(f.numpy())
            writer.close()
    out = torch.stack(out, axis=0)
    return out

def demo_draw_to_batch(smpl_joints_batch, title_batch=None, outname=None) : 
    
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out

def mask_png(frames):
    for frame in frames:
        im = imageio.imread(frame)
        if im.shape[-1] > 3:
            im[im[:, :, 3] < 1, :] = 255
            imageio.imwrite(frame, im[:, :, 0:3])
    return


class Video:
    def __init__(self, frame_path: str, fps: float = 12.5, res="high"):
        frame_path = str(frame_path)
        self.fps = fps

        self._conf = {"codec": "libx264",
                      "fps": self.fps,
                      "audio_codec": "aac",
                      "temp_audiofile": "temp-audio.m4a",
                      "remove_temp": True}

        if res == "low":
            bitrate = "500k"
        else:
            bitrate = "5000k"

        self._conf = {"bitrate": bitrate,
                      "fps": self.fps}

        frames = [os.path.join(frame_path, x)
                  for x in sorted(os.listdir(frame_path))]

        # mask background white for videos
        mask_png(frames)

        video = mp.ImageSequenceClip(frames, fps=fps)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        # needs ImageMagick
        video_text = mp.TextClip(text,
                                 font='Amiri',
                                 color='white',
                                 method='caption',
                                 align="center",
                                 size=(self.video.w, None),
                                 fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        # video_text = video_text.set_pos('bottom')
        video_text = video_text.set_pos('top')

        self.video = mp.CompositeVideoClip([self.video, video_text])

    def save(self, out_path):
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(
            out_path, **self._conf)
        
def save_blender_result_video(paths):
    pbar = tqdm(paths)
    for path in pbar:
        pbar.set_description(f"processing {path}")
        if os.path.exists(path.replace(".npy", ".mp4")):
            print(f"npy is rendered or under rendering {path}")
            continue

        frames_folder = os.path.join(path.replace("mesh.npy", "result"), "frames")

        if os.path.isdir(frames_folder):
            video = Video(frames_folder, fps=20)

            vid_path = frames_folder.replace("frames", "video.mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=str, default="blender", choices=["blender", "mesh", 'joints'])
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--dir', type=str)
    
    arg = parser.parse_args()

    output_dir = arg.dir
    paths = []
    # paths.append()
    file_list = natsort.natsorted(os.listdir(output_dir))
    for item in file_list:
        if item.endswith("mesh.npy"):
            paths.append(os.path.join(output_dir, item))

    save_blender_result_video(paths)
        
    