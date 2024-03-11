import os 
import numpy as np
import cv2
from pathlib import Path 
import scipy
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def compute_vec(X, Y, flow_dir, kp, mask):
    flow = scipy.io.loadmat(str(flow_dir/f'{n:05d}_denseflow.mat'))['flow']
    # flow[mask==0] = 0
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


def compute_vec_err(vec1, vec2, rm_boundary=False):
    if rm_boundary:
        W = 15
        H = 13
        L = []
        for j in np.arange(1, H-1):
            for i in np.arange(1, W-1):
                L.append(j*W + i)

        vec1 = vec1[L, :]
        vec2 = vec2[L, :]
        
    err_list = np.sqrt((vec1[:,0] - vec2[:,0]) ** 2 + (vec1[:,1] - vec2[:,1]) ** 2)
    return err_list

thing = "rope"
ID_list = [14,15,16,17,18,19]
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

        err_squence_our = np.zeros((N, 1))
        err_squence_pwc = np.zeros((N, 1))
        err_squence_of = np.zeros((N, 1))
        err_squence_raft = np.zeros((N, 1))
        err_squence_dalf = np.zeros((N, 1))
        err_squence_omni = np.zeros((N, 1))
        for n in range(N-1):
            print(f'{thing}{ID}:{cam}:{n}/{N}')

            kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))
            vec_gt = np.float32(np.loadtxt(str(our_dir / f'{n:05d}_gridflow.txt'), dtype=float))    

            im1 = cv2.imread(str(im_paths[n]))
            H, W = im1.shape[:2]

            # mask1 = cv2.imread(str(mask_paths[n]))[:,:,0]
            mask1 = []

            X = np.arange(0, W) + 0.5
            Y = np.arange(0, H) + 0.5
            
            vec_our = compute_vec(X, Y, our_dir, kp1, mask1)
            vec_pwc = compute_vec(X, Y, pwc_dir, kp1, mask1)
            vec_of = compute_vec(X, Y, of_dir, kp1, mask1)
            vec_raft = compute_vec(X, Y, raft_dir, kp1, mask1)
            vec_dalf = compute_vec(X, Y, dalf_dir, kp1, mask1)
            vec_omni = compute_vec(X, Y, omni_dir, kp1, mask1)

            # plot_arrow(im1, kp1, vec_pwc)
            # exit()
            err_all_our = compute_vec_err(vec_gt, vec_our)
            err_all_pwc = compute_vec_err(vec_gt, vec_pwc)
            err_all_of = compute_vec_err(vec_gt, vec_of)
            err_all_raft = compute_vec_err(vec_gt, vec_raft)
            err_all_dalf = compute_vec_err(vec_gt, vec_dalf)
            err_all_omni = compute_vec_err(vec_gt, vec_omni)

            # plt.figure()
            # plt.plot(err_all_our, 'k')
            # plt.plot(err_all_pwc, 'r')
            # plt.plot(err_all_of, 'g')
            # plt.plot(err_all_raft, 'b')
            # plt.show()
            # exit()

            err_squence_our[n] = np.mean(err_all_our)
            err_squence_pwc[n] = np.mean(err_all_pwc)
            err_squence_of[n] = np.mean(err_all_of)
            err_squence_raft[n] = np.mean(err_all_raft)
            err_squence_dalf[n] = np.mean(err_all_dalf)
            err_squence_omni[n] = np.mean(err_all_omni)


        f = open("err.txt","a")
        f.write(f'{thing}{ID}:cam0{cam}:error_our:{np.mean(err_squence_our):.3f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_of:{np.mean(err_squence_of):.3f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_pwc:{np.mean(err_squence_pwc):.3f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_raft:{np.mean(err_squence_raft):.3f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_omni:{np.mean(err_squence_omni):.3f}\n')
        f.write(f'{thing}{ID}:cam0{cam}:error_dalf:{np.mean(err_squence_dalf):.3f}\n')
        f.close()

        # plt.figure()
        # plt.plot(err_squence_our, 'k')
        # plt.plot(err_squence_pwc, 'r')
        # plt.plot(err_squence_of, 'g')
        # plt.plot(err_squence_raft, 'b')
        # plt.show()
    
        # exit()
