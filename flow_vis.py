# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np
import torch
import cv2
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np
import torch.nn.functional as F


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def sparse_to_dense(coord_xy, flow_xy, size):
    # coord_xy, N by 2 array
    # flow_xy, N by 2 array
    # size, (W, H)
    # return flow, H by W by 2 array

    interp = LinearNDInterpolator(coord_xy, flow_xy, fill_value=0)
    W, H = size
    X = np.arange(0, W)
    Y = np.arange(0, H)
    X, Y = np.meshgrid(X, Y)
    flow = np.float32(interp(X, Y))

    return flow


def save_fig(save_path, im, kp, size=None):
    plt.figure(figsize=(20,16))
    plt.imshow(im[...,::-1])
    if size is not None:    
        W, H = size
        for i in range(H):
            plt.plot(kp[W*i:W*(i+1),0], kp[W*i:W*(i+1),1], color=(0,1,0))
        for i in range(W):
            plt.plot(kp[i::W,0], kp[i::W,1], color=(0,1,0))
    else:
        plt.plot(kp[:,0], kp[:,1], color=(0,1,0))

    plt.scatter(kp[:,0], kp[:,1], color=(0,1,0))
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def warp_image_cv(img, kp1, kp2, device):
    h, w = img.shape[:2]
    img = torch.tensor(img).to(device).permute(2,0,1)[None, ...].float()

    c_src = kp1 / np.float32([w, h])
    c_dst = kp2 / np.float32([w, h]) 
    theta = tps_np.tps_theta_from_points(c_src, c_dst, reduced=True, lambd=0.01)
    theta = torch.tensor(theta).to(device)[None, ...]
    grid = tps_pth.tps_grid(theta, torch.tensor(c_dst, device=device), img.shape)

    img = F.grid_sample(img, grid, align_corners = True, padding_mode = 'border')
    return img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)


def warp_image_dense(im, flow):
    H, W = im.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(W)
    flow[:,:,1] += np.arange(H)[:,np.newaxis]
    return cv2.remap(im, flow, None, interpolation=cv2.INTER_LINEAR, borderValue=(0,0,0))


    # ffmpeg -f image2 -framerate 12 -i foo-%03d.jpeg -s WxH foo.avi