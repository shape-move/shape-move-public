import os
import random
import shutil
import sys
from pathlib import Path
import argparse
from argparse import ArgumentParser
import numpy as np

sys.path.insert(0, f'{os.path.join(os.path.dirname(__file__))}/..')
from utils.blender.camera import Camera
from utils.blender.floor import plot_floor, show_traj
from utils.blender.sampler import get_frameidx
from utils.blender.scene import setup_scene 
from utils.blender.tools import delete_objs, style_detect

try:
    import bpy

    sys.path.append(os.path.dirname(bpy.data.filepath))

    # local packages
    sys.path.append(os.path.expanduser("~/.local/lib/python3.9/site-packages"))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )


def parse_args(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)


setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_args)

def render_cli(arg) -> None:
    output_dir = arg.dir
    paths = []
    file_list = os.listdir(output_dir)

    for item in file_list:
        if item.endswith("_mesh.npy"):
            paths.append(os.path.join(output_dir, item))

    paths = paths
    init = True
    for path in paths:
        video_folder = os.path.join(path.replace("mesh.npy", "result"), "video.mp4")
        if os.path.isfile(video_folder):
            print(f"npy is rendered or under rendering {path}")
            continue

        frames_folder = os.path.join(path.replace("mesh.npy", "result"), "frames")
        os.makedirs(frames_folder, exist_ok=True)

        try:
            data = np.load(path)
            if data.shape[0] == 1:
                data = data[0]
        except FileNotFoundError:
            print(f"{path} not found")
            continue

        render(
            data,
            frames_folder,
            canonicalize=True,
            exact_frame=0.5,
            num=50,
            mode=arg.mode,
            model_path=None,
            faces_path='data/smpl/smpl.faces',
            downsample=False,
            always_on_floor=False,
            oldrender=True,
            res='high',
            init=init,
            gt=False,
            accelerator='gpu',
            device=[0],
        )

        init = False

        if os.path.exists(frames_folder+'_img'):
            shutil.rmtree(frames_folder+'_img')
        shutil.copytree(frames_folder, frames_folder+'_img') 

def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)

def render(npydata,
           frames_folder,
           *,
           mode,
           model_path,
           faces_path,
           gt=False,
           exact_frame=None,
           num=8,
           downsample=True,
           canonicalize=True,
           always_on_floor=False,
           denoising=True,
           oldrender=True,
           res="high",
           init=True,
           accelerator='gpu',
           device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res,
                    denoising=denoising,
                    oldrender=oldrender,
                    accelerator=accelerator,
                    device=device)

    is_mesh, is_smplx, jointstype = style_detect(npydata)


    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"


    if is_mesh:
        from utils.blender.meshes import Meshes
        data = Meshes(npydata,
                      gt=gt,
                      mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      is_smplx=is_smplx)
    else:
        from utils.blender.joints import Joints
        data = Joints(npydata,
                      gt=gt,
                      mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    plot_floor(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidx = get_frameidx(mode=mode,
                            nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render - 1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render - 1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    bpy.ops.wm.save_as_mainfile(filepath=frames_folder.replace('.png','.blend').replace('frames','blender.blend'))

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument('--num', nargs='+', default=None)

    arg = parser.parse_args()

    print(arg)

    render_cli(arg)

