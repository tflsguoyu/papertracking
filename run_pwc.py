# Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunnar Farneback in 2003.
import os
import sys
sys.path.append("DALF_CVPR_2023")
import numpy as np
import cv2
from pathlib import Path
import flow_vis
import scipy

thing = "rope"
ID_list = [14,15,16,17,18,19]
cam_list = [32]
for ID in ID_list:
    for cam in cam_list:

        in_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}")
        out_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}/pwc")
        out_dir.mkdir(exist_ok=True)

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))

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

            os.chdir('C:/Users/guoyu/cvpr/PWC-Net/PyTorch/')
            os.system(f'python script_pwc.py {str(im_paths[n])} {str(im_paths[n+1])} {str(out_dir / (fn + "_denseflow.mat"))}')

            flow = scipy.io.loadmat(str(out_dir / (fn + "_denseflow.mat")))
            flow = flow['flow']
            print(flow.shape)

            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
            cv2.imwrite(str(out_dir / (fn + "_denseflow.jpg")), flow_color)    

            im1_warped_dense = flow_vis.warp_image_dense(im1, flow)
            cv2.imwrite(str(out_dir / (fn + "_warped_dense.jpg")), im1_warped_dense)

            # exit()