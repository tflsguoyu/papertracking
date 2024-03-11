import os 
import numpy as np
import cv2
from pathlib import Path 
import scipy
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import time


def compute_vec(X, Y, flow_dir, kp, mask):
    flow = scipy.io.loadmat(str(flow_dir/f'{n:05d}_denseflow.mat'))['flow']
    interp = RegularGridInterpolator((X, Y), np.transpose(flow, (1,0,2)))
    return interp(kp)

def plot_arrow(im, kp, vec):
    plt.figure()
    plt.imshow(im[...,::-1])
    # plt.arrow(kp[:,0], kp[:,1], vec[:,0], vec[:,1])
    for i in range(kp.shape[0]):
        plt.arrow(kp[i,0], kp[i,1], vec[i,0], vec[i,1], color='green', length_includes_head=True, head_width=5)
    plt.axis('off')
    plt.show()


def compute_RSME(im1, im2, mask):
    # mask = np.dstack([mask, mask, mask])
    # im1 = im1[mask==255].astype(np.float32)/255
    # im2 = im2[mask==255].astype(np.float32)/255
    im1 = im1.flatten().astype(np.float32)/255
    im2 = im2.flatten().astype(np.float32)/255

    # rsme1 = np.sqrt(np.mean((im1 - im2)**2))
    rsme2 = np.linalg.norm(im1 - im2) / np.sqrt(len(im1))
    return rsme2

thing = "cloth"
ID_list = [2,3,4,5,9,10]
cam_list = [32]
root_dir = Path("D:/data")



for ID in ID_list:
    for cam in cam_list:

        in_dir = root_dir / f"{thing}{ID}/cam0{cam}"
        our_dir = in_dir / "ours"
        pwc_dir = in_dir / "pwc"
        of_dir = in_dir / "opencvof"
        raft_dir = in_dir / "raft"
        dalf_dir = in_dir / "dalf"
        omni_dir = in_dir / "omni"

        im_paths = sorted(list((in_dir/'color').glob("[!.]*.jpg")))
        kp_paths = sorted(list((in_dir/'gt_marker').glob("[!.]*.txt")))
        mask_paths = sorted(list((in_dir/'mask').glob("[!.]*.png")))

        N = len(im_paths)

        err_squence_our = np.zeros((N-1, 1))
        err_squence_pwc = np.zeros((N-1, 1))
        err_squence_of = np.zeros((N-1, 1))
        err_squence_raft = np.zeros((N-1, 1))
        err_squence_dalf = np.zeros((N-1, 1))
        err_squence_omni = np.zeros((N-1, 1))
        for n in range(N-1):
            print(f'{thing}{ID}:{cam}:{n}/{N}')

            kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))
            vec_gt = np.float32(np.loadtxt(str(our_dir / f'{n:05d}_gridflow.txt'), dtype=float))    

            im1 = cv2.imread(str(im_paths[n]))
            H, W = im1.shape[:2]

            # mask1 = cv2.imread(str(mask_paths[n]))[:,:,0]
            mask1 = []

            im2 = cv2.imread(str(im_paths[n+1]))
            # mask2 = cv2.imread(str(mask_paths[n+1]))[:,:,0]
            mask2 = []

            X = np.arange(0, W) + 0.5
            Y = np.arange(0, H) + 0.5

            im1_warped_our = cv2.imread(str(our_dir/f'{n:05d}_warped.jpg'))            
            im1_warped_pwc = cv2.imread(str(pwc_dir/f'{n:05d}_warped_dense.jpg'))            
            im1_warped_of = cv2.imread(str(of_dir/f'{n:05d}_warped_dense.jpg'))            
            im1_warped_raft = cv2.imread(str(raft_dir/f'{n:05d}_warped_dense.jpg'))            
            im1_warped_dalf = cv2.imread(str(dalf_dir/f'{n:05d}_warped.jpg'))   
            im1_warped_omni = cv2.imread(str(omni_dir/f'{n:05d}_warped_dense.jpg'))            
      


            err_squence_our[n] = compute_RSME(im2, im1_warped_our, mask2)
            err_squence_pwc[n] = compute_RSME(im2, im1_warped_pwc, mask2)
            err_squence_of[n] = compute_RSME(im2, im1_warped_of, mask2)
            err_squence_raft[n] = compute_RSME(im2, im1_warped_raft, mask2)
            err_squence_dalf[n] = compute_RSME(im2, im1_warped_dalf, mask2)
            err_squence_omni[n] = compute_RSME(im2, im1_warped_omni, mask2)


        f = open("err2.txt","a")
        f.write(f'{thing}{ID}:cam0{cam}:error_our:{255*np.mean(err_squence_our):.2f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_of:{255*np.mean(err_squence_of):.2f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_pwc:{255*np.mean(err_squence_pwc):.2f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_raft:{255*np.mean(err_squence_raft):.2f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_omni:{255*np.mean(err_squence_omni):.2f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_dalf:{255*np.mean(err_squence_dalf):.2f}\n')
        f.close()

        # plt.figure()
        # plt.plot(err_squence_our, 'k')
        # plt.plot(err_squence_pwc, 'r')
        # plt.plot(err_squence_of, 'g')
        # plt.plot(err_squence_raft, 'b')
        # plt.show()
    
        # exit()
