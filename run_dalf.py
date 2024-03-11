import sys
sys.path.append("DALF_CVPR_2023")
from modules.models.DALF import DALF_extractor as DALF
import flow_vis
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from pathlib import Path
from modules.tps import RANSAC
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np
import torch.nn.functional as F

from matplotlib import cm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

thing = "rope"
ID_list = [14,15,16,17,18,19]
cam_list = [32]
for ID in ID_list:
    for cam in cam_list:

        in_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}")
        out_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}/dalf")
        out_dir.mkdir(exist_ok=True)

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))


        dalf = DALF(dev = device)

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

            #Compute kps and features
            dalf_kp1, descs1 = dalf.detectAndCompute(im1)
            dalf_kp2, descs2 = dalf.detectAndCompute(im2)

            # flow_vis.save_fig(str(out_dir / (fn + "_kp_tmp.jpg")), im1, np.float32(dalf_kp1))

            #Match using vanilla opencv matcher
            matcher = cv2.BFMatcher(crossCheck = True)
            matches = matcher.match(descs1, descs2)

            src_pts = np.float32([dalf_kp1[m.queryIdx].pt for m in matches])
            tgt_pts = np.float32([dalf_kp2[m.trainIdx].pt for m in matches])

            #Computes non-rigid RANSAC
            inliers = RANSAC.nr_RANSAC(src_pts, tgt_pts, device, thr = 0.2)
            good_matches = [matches[i] for i in range(len(matches)) if inliers[i]]

            dalf_filtered_kp1 = np.float32([dalf_kp1[m.queryIdx].pt for m in good_matches])
            dalf_filtered_kp2 = np.float32([dalf_kp2[m.trainIdx].pt for m in good_matches])

            np.savetxt(str(out_dir / (fn + "_kp.txt")), dalf_filtered_kp1, delimiter=' ', fmt='%.4f')

            dalf_filtered_kp_vec = dalf_filtered_kp2 - dalf_filtered_kp1
            np.savetxt(str(out_dir / (fn + "_sparseflow.txt")), dalf_filtered_kp_vec, delimiter=' ', fmt='%.4f')

            flow_vis.save_fig(str(out_dir / (fn + "_kp.jpg")), im1, dalf_filtered_kp1)

            im1_warped = flow_vis.warp_image_cv(im1, dalf_filtered_kp1, dalf_filtered_kp2, device)
            cv2.imwrite(str(out_dir / (fn + "_warped.jpg")), im1_warped)


            H, W = im1.shape[:2]
            flow = flow_vis.sparse_to_dense(dalf_filtered_kp1, dalf_filtered_kp_vec, (W, H))
            flow_mdic = {"flow": flow}
            scipy.io.savemat(str(out_dir / (fn + "_denseflow.mat")), flow_mdic)

            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
            cv2.imwrite(str(out_dir / (fn + "_denseflow.jpg")), flow_color)

            im1_warped_dense = flow_vis.warp_image_dense(im1, flow)
            cv2.imwrite(str(out_dir / (fn + "_warped_dense.jpg")), im1_warped_dense)

            # exit()
