import os
import sys
sys.path.append("DALF_CVPR_2023")
import torch
import cv2
import numpy as np
from pathlib import Path
import scipy
from matplotlib import cm
from PIL import ImageEnhance, Image
import flow_vis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

color_map = cm.get_cmap("jet")

thing = "rope"
ID_list = [14,15,16,17,18,19]
cam_list = [32]

for ID in ID_list:
    for cam in cam_list:

        in_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}")
        out_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}/ours")
        out_dir.mkdir(exist_ok=True)

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))


        N = len(im_paths)
  
        for n in range(N):
            fn = im_paths[n].stem
            print(f"{thing}{ID}:{cam}:{fn}")

            im1 = cv2.imread(str(im_paths[n]))
            kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))

            if thing == "paper":
                flow_vis.save_fig(str(out_dir / (fn + "_kp.jpg")), im1, kp1, size=(15,13))
            if thing == "cloth":
                flow_vis.save_fig(str(out_dir / (fn + "_kp.jpg")), im1, kp1, size=(15,15))
            if thing == "rope":
                flow_vis.save_fig(str(out_dir / (fn + "_kp.jpg")), im1, kp1)

            if n == N-1:
                break 

            im2 = cv2.imread(str(im_paths[n+1]))
            kp2 = np.float32(np.loadtxt(str(kp_paths[n+1]), dtype=float))

            # 
            im1_warped = flow_vis.warp_image_cv(im1, kp1, kp2, device)
            cv2.imwrite(str(out_dir / (fn + "_warped.jpg")), im1_warped)

            kp_vec = kp2 - kp1
            np.savetxt(str(out_dir / (fn + "_gridflow.txt")), kp_vec, delimiter=' ', fmt='%.4f')

            H, W = im1.shape[:2]
            flow = flow_vis.sparse_to_dense(kp1, kp_vec, (W, H))
            flow_mdic = {"flow": flow}
            scipy.io.savemat(str(out_dir / (fn + "_denseflow.mat")), flow_mdic)

            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
            cv2.imwrite(str(out_dir / (fn + "_denseflow.jpg")), flow_color)

            im1_warped_dense = flow_vis.warp_image_dense(im1, flow)
            cv2.imwrite(str(out_dir / (fn + "_warped_dense.jpg")), im1_warped_dense)

        os.system(f'ffmpeg -i {out_dir}/%05d_kp.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {out_dir}/kp.mp4')