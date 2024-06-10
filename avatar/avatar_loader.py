
import os
import cv2
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import collections
import struct
import json
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from avatar.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params,
    get_static_camera
)
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def camera_info(path,file,white_background,extension):
    cam_infos = []
    file = os.path.join(path, file)

    with open(file, 'rb') as f:
        datas = json.load(f)
        for idx, data in enumerate(datas):
            print(data)
            cam_name = os.path.join(path, "blender",data["img_id"] + extension)
            mask_name = os.path.join(path, "mask",data["img_id"]+"_mask"+extension)
            c2w = np.array(data["extrinsics"]["c2w_matrix"])
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            K = np.array(data["intrinsics"])

            FovX = 2 * np.arctan(data["width"] / (2 * K[0, 0]))
            FovY = 2 * np.arctan(data["height"] / (2 * K[1, 1]))

            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255
            mask.astype(np.float32)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")



            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask = mask,
                                        image_path=cam_name, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

        return cam_infos


if __name__ == '__main__':
    path = "/usr/local/gaussian-splatting/humans"
    file = "cameras.json"
    extension = ".png"
    camera_info(path,file,None,extension)
