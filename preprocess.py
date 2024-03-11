import os 
from pathlib import Path
import cv2
import numpy as np


# paper: padding = 10, line 92: if n % 5 == 0:
# rope : padding = 25, line 92: if n % 1 == 0:

def remove_boundary(im, kp, padding=25):
    H, W = im.shape[:2]
    im[:max(0, int(min(kp[:,1]))-padding), :] = 255
    im[min(H, int(max(kp[:,1]))+padding):, :] = 255
    im[:, :max(0, int(min(kp[:,0]))-padding)] = 255
    im[:, min(W, int(max(kp[:,0]))+padding):] = 255
    return im


cam_list = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
for cam in cam_list:

    in_dir = Path(f"C:/Users/guoyu/Downloads/1_ref/1/raw/cam0{cam}/raw")
    out_dir = Path(f"C:/Users/guoyu/data/rope1/cam0{cam}/color")
    out_dir.mkdir(parents=True, exist_ok=True)

    im_paths = sorted(list(in_dir.glob("[!.]*.png")))
    kp_paths = sorted(list(in_dir.glob("[!.]*.txt")))


    N = len(im_paths)
    H = 2048
    W = 2448
    top = 99999
    bot = 0
    left = 99999
    right = 0
    pad = 10

    for n in range(N):
        kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))

        left_this = min(kp1[:,0])
        right_this = max(kp1[:,0])
        top_this = min(kp1[:,1])
        bot_this = max(kp1[:,1])

        if left_this < left:
            left = left_this

        if right_this > right:
            right = right_this

        if top_this < top:
            top = top_this

        if bot_this > bot:
            bot = bot_this

    left = max(int(left - pad), 0)
    right = min(int(right + pad), W)
    top = max(int(top - pad), 0)
    bot = min(int(bot + pad), H)        

    print(left, right, top, bot)

    H_extra = 8 - (bot - top) % 8
    top -= (H_extra // 2)
    bot += (H_extra - H_extra // 2) 

    if top < 0:
        bot += top
        top -= top
    if bot > H:
        top += (bot - H)
        bot -= (bot - H)

    W_extra = 8 - (right - left) % 8
    left -= (W_extra // 2)
    right += (W_extra - W_extra // 2) 

    if left < 0:
        right += left
        left -= left
    if right > W:
        left += (right - W)
        right -= (right - W)

    print(left, right, top, bot)

    idx = 0
    for n in range(N):
        if n % 1 == 0:
            fn = im_paths[n].stem

            im1 = cv2.imread(str(im_paths[n]))
            kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))

            kp1[:, 0] -= (left+1.5)
            kp1[:, 1] -= (top+1.5)

            im1_crop = im1[top:bot, left:right, :]
            im1_final = remove_boundary(im1_crop, kp1)

            cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), im1_final)
            np.savetxt(str(out_dir / f"{idx:05d}.txt"), kp1, delimiter=' ', fmt='%.4f')
            
            idx += 1

    os.system(f'ffmpeg -i {out_dir}/%05d.jpg {out_dir}/video.mp4')
