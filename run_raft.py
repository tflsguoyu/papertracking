import sys
sys.path.append('RAFT/core')
sys.path.append("DALF_CVPR_2023")
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from pathlib import Path
import scipy

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import flow_vis

from matplotlib import cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

thing = "rope"
ID_list = [14,15,16,17,18,19]
cam_list = [32]
for ID in ID_list:
    for cam in cam_list:

        in_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}")
        out_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}/raft")
        out_dir.mkdir(exist_ok=True)

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))


        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="RAFT/models/raft-things.pth")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()

        with torch.no_grad():
            N = len(im_paths)

            for n in range(N):
                fn = im_paths[n].stem
                print(fn)

                im1 = cv2.imread(str(im_paths[n]))
                kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))

                if n == N-1:
                    break 

                im2 = cv2.imread(str(im_paths[n+1]))
                kp2 = np.float32(np.loadtxt(str(kp_paths[n+1]), dtype=float))

        
                im1_th = torch.from_numpy(im1).permute(2, 0, 1).float()[None].to(device)
                im2_th = torch.from_numpy(im2).permute(2, 0, 1).float()[None].to(device)
                
                padder = InputPadder(im1_th.shape, 'thing')
                im1_th, im2_th = padder.pad(im1_th, im2_th)

                flow_low, flow = model(im1_th, im2_th, iters=20, test_mode=True)
                flow = flow[0].permute(1,2,0).cpu().numpy()

                if thing == "rope":
                    flow = flow[:, :612, :]

                flow_mdic = {"flow": flow}
                scipy.io.savemat(str(out_dir / (fn + "_denseflow.mat")), flow_mdic)

                flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
                cv2.imwrite(str(out_dir / (fn + "_denseflow.jpg")), flow_color)

                im1_warped_dense = flow_vis.warp_image_dense(im1, flow)
                cv2.imwrite(str(out_dir / (fn + "_warped_dense.jpg")), im1_warped_dense)

