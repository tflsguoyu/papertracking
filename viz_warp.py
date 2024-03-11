import sys
sys.path.append("DALF_CVPR_2023")
from modules.models.DALF import DALF_extractor as DALF
import flow_vis

import os 
import numpy as np
import cv2
from pathlib import Path 
import scipy
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import ImageEnhance, Image

color_map = cm.get_cmap("jet")

def load_image(im_paths, f1, f2):
    im1 = cv2.imread(str(im_paths[f1]))
    im2 = cv2.imread(str(im_paths[f2]))
    return im1, im2

def load_kp(kp_paths, f, rm_boundary=False):
    kp = np.float32(np.loadtxt(str(kp_paths[f]), dtype=float))
    
    if rm_boundary:
        L = []; H = 13; W = 15
        for j in np.arange(1, H-1):
            for i in np.arange(1, W-1):
                L.append(j*W + i)
        kp = kp[L, :]
    
    return kp


def get_flow(method_dir, f, mask_paths):
    flow = scipy.io.loadmat(str(method_dir/f'{f:05d}_denseflow.mat'))['flow']
    return flow

def get_next_kp(flow, kp):
    H, W = flow.shape[:2]
    X = np.arange(0, W) + 0.5
    Y = np.arange(0, H) + 0.5
    interp = RegularGridInterpolator((X, Y), np.transpose(flow, (1,0,2)))
    kp_new = kp + np.float32(interp(kp))
    kp_new[kp_new < 0] = 0.5
    x = kp_new[:, 0]
    x[x > W] = W - 0.5
    y = kp_new[:, 1]
    y[y > H] = H - 0.5
    kp_new[:, 0] = x
    kp_new[:, 1] = y
    return kp_new


def track_kps(kp_paths, mask_paths, data_dir, method, f1, f2):
    method_dir = data_dir / method
    kps = []
    kps.append(load_kp(kp_paths, f1))
    for f in np.arange(f1, f2):
        # get kp of f1+1 
        if method == "gt":
            kps.append(load_kp(kp_paths, f+1))
        else:
            flow = get_flow(method_dir, f, mask_paths)
            kp_next = get_next_kp(flow, kps[-1])
            kps.append(kp_next)

    return kps

def draw_trail(im, kps):
    converter = ImageEnhance.Color(Image.fromarray(im))
    im = np.array(converter.enhance(0.5))

    idx = 0
    for kp1 in kps[:-1]:
        idx += 1
        kp2 = kps[idx]
        num_pts = kp1.shape[0]

        for i in range(num_pts):
            color = np.array(color_map(i/max(1, float(num_pts - 1)))[:3]) * 255
            p1 = (int(round(kp1[i,0])), int(round(kp1[i,1])))
            p2 = (int(round(kp2[i,0])), int(round(kp2[i,1])))
            cv2.line(im, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)

    for j in range(num_pts):
        color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
        p2 = (int(round(kp2[j,0])), int(round(kp2[j,1])))
        cv2.circle(im, p2, 2, color, -1, lineType=16)

    return im


def draw_marker(im, kp):
    # converter = ImageEnhance.Color(Image.fromarray(im))
    # im = np.array(converter.enhance(0.5))
    num_pts = kp.shape[0]
    for j in range(num_pts):
        color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
        p = (int(round(kp[j,0])), int(round(kp[j,1])))
        cv2.circle(im, p, 2, color, -1, lineType=16)

    return im


def warp_image(im, kps):
    im_warped = im.copy()
    idx = 0
    for kp in kps[:-1]:
        idx += 1
        kp2 = kps[idx]
        im_warped = flow_vis.warp_image_cv(im_warped, kp, kp2, "cuda")
    return im_warped

def main(root_dir):
    thing = "rope"
    ID_list = [18]
    cam_list = [32]

    # method = "gt"
    # method = "ours"
    # method = "raft"
    # method = "dalf"
    method = "omni"
    # method = "opencvof"
    # method = "pwc"

    start = 0
    end_list = [25]
    # end_list = np.arange(50,199)

    for ID in ID_list:
        for cam in cam_list:
            data_dir = root_dir / f"{thing}{ID}/cam0{cam}"

            im_paths = sorted(list((data_dir/'color').glob("[!.]*.jpg")))
            kp_paths = sorted(list((data_dir/'gt_marker').glob("[!.]*.txt")))
            # mask_paths = sorted(list((data_dir/'mask').glob("[!.]*.png")))
            mask_paths = []

            method_dir = data_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            idx = 0
            for end in end_list: 
                print(end)
                im_start, im_end = load_image(im_paths, start, end)
                
                if start == end:
                    kps = load_kp(kp_paths, start)
                    im_trail = draw_marker(im_start, kps)
                    cv2.imwrite(str(method_dir / f"{idx:05d}.png"), im_trail)
                else:
                    kps = track_kps(kp_paths, mask_paths, data_dir, method, start, end)
                    if method == "gt":
                        im_trail = draw_trail(im_end, kps)
                    else:
                        im_warped = warp_image(im_start, kps)
                        im_trail = draw_trail(im_warped, kps)
                    cv2.imwrite(str(method_dir / f"{idx:05d}.png"), im_trail)

                idx += 1
            # os.system(f'ffmpeg -i {method_dir}/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {method_dir}/viz.mp4')

if __name__ == "__main__":
    main(Path("D:/data"))


