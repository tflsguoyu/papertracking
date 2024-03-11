import sys
sys.path.append("DALF_CVPR_2023")
import numpy as np
import cv2
from pathlib import Path
import scipy
from matplotlib import cm
import flow_vis

thing = "rope"
ID_list = [14,15,16,17,18,19]
cam_list = [32]
for ID in ID_list:
    for cam in cam_list:

        in_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}")
        out_dir = Path(f"D:/data/{thing}{ID}/cam0{cam}/omni")
        out_dir.mkdir(exist_ok=True)

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))

        flow_dir = Path(f'C:/Users/guoyu/cvpr/omnimotion/out/{thing}{ID}_cam0{cam}/flow')
        flow_paths = sorted(list(flow_dir.glob("[!.]*.npy")))

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

            flow = np.load(flow_paths[n])
            # if cam == 32:
	        #     flow = scipy.ndimage.zoom(flow, (2,2,1)) * 2

            flow_mdic = {"flow": flow}
            scipy.io.savemat(str(out_dir / (fn + "_denseflow.mat")), flow_mdic)

            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
            cv2.imwrite(str(out_dir / (fn + "_denseflow.jpg")), flow_color)

            im1_warped_dense = flow_vis.warp_image_dense(im1, flow)
            cv2.imwrite(str(out_dir / (fn + "_warped_dense.jpg")), im1_warped_dense)
