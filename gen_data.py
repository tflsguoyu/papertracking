import os
import numpy as np
from pathlib import Path
import cv2

def find_box(kp_paths, image_size, pad=10):
    N = len(kp_paths)
    H, W = image_size

    top = 99999
    bot = 0
    left = 99999
    right = 0
  
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

    return left, right, top, bot


def refine_box(left, right, top, bot, image_size, res=8):
    H, W = image_size
    H_extra = res - (bot - top) % res
    top -= (H_extra // 2)
    bot += (H_extra - H_extra // 2) 

    if top < 0:
        bot += top
        top -= top
    if bot > H:
        top += (bot - H)
        bot -= (bot - H)

    W_extra = res - (right - left) % res
    left -= (W_extra // 2)
    right += (W_extra - W_extra // 2) 

    if left < 0:
        right += left
        left -= left
    if right > W:
        left += (right - W)
        right -= (right - W)

    return left, right, top, bot

#### Main
# thing = "paper"
thing = "cloth"
# thing = "rope"

ID_list = [100]
cam_list = [32]
pick_frame = 1


if thing == "rope":
    in_dir = Path('D:/raw/ropes')
    out_dir = Path('D:/data')

    for ropeID in ID_list:
        out_paper_dir = out_dir / f'rope{ropeID}'
        out_paper_dir.mkdir(parents=True, exist_ok=True)
        for cam in cam_list:
            out_cam_dir = out_paper_dir / f'cam0{cam}'
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            
            # Video to images
            video_dir = in_dir / f'{ropeID}/ref/cam0{cam}'
            video_dir.mkdir(parents=True, exist_ok=True)
            cmd = f'ffmpeg -i {video_dir}.mp4 {video_dir}/%05d.png'
            print(cmd)
            os.system(cmd)
            os.remove(f'{video_dir}/00001.png')

            im_paths = sorted(list(video_dir.glob("[!.]*.png")))
            H, W = cv2.imread(str(im_paths[0])).shape[:2]
            kp_dir = in_dir / f'{ropeID}/markers2d_ref'
            kp_paths = sorted(list(kp_dir.glob(f"[!.]*{cam}.txt")))
           
            out_frame_dir = out_cam_dir / 'color'
            out_frame_dir.mkdir(parents=True, exist_ok=True)
            out_marker_dir = out_cam_dir / 'gt_marker'
            out_marker_dir.mkdir(parents=True, exist_ok=True)

            idx = 0
            for n in range(len(im_paths)):
                if n % pick_frame == 0:
                    fn = im_paths[n].stem

                    im1 = cv2.imread(str(im_paths[n]))
                    kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))
                    kp1 /= 4

                    cv2.imwrite(str(out_frame_dir / f"{idx:05d}.jpg"), cv2.resize(im1, (int(W/4), int(H/4))))
                    # cv2.imwrite(str(out_frame_dir / f"{idx:05d}.jpg"), im1)
                    np.savetxt(str(out_marker_dir / f"{idx:05d}.txt"), kp1, delimiter=' ', fmt='%.4f')
                    
                    idx += 1

if thing == "paper":
    in_dir = Path('C:/Users/guoyu/Downloads/papers')
    out_dir = Path('C:/Users/guoyu/data_new')

    for paperID in ID_list:
        out_paper_dir = out_dir / f'paper{paperID}'
        out_paper_dir.mkdir(parents=True, exist_ok=True)
        for cam in cam_list:
            out_cam_dir = out_paper_dir / f'cam0{cam}'
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            
            # Video to images
            video_dir = in_dir / f'{paperID}/ref/cam0{cam}'
            video_dir.mkdir(parents=True, exist_ok=True)
            cmd = f'ffmpeg -i {video_dir}.mp4 {video_dir}/%05d.png'
            print(cmd)
            os.system(cmd)
            os.remove(f'{video_dir}/00001.png')

            # Check images
            im_paths = sorted(list(video_dir.glob("[!.]*.png")))
            kp_dir = in_dir / f'{paperID}/markers2d_ref'
            kp_paths = sorted(list(kp_dir.glob(f"[!.]*{cam}.txt")))
            mask_dir = in_dir / f'{paperID}/template_mask'
            mask_paths = sorted(list(mask_dir.glob(f"[!.]*{cam}.png")))

            # Find bouding box
            H, W = cv2.imread(str(im_paths[0])).shape[:2]
            left, right, top, bot = find_box(kp_paths, (H, W))
            print(left, right, top, bot)
            left, right, top, bot = refine_box(left, right, top, bot, (H, W))
            print(left, right, top, bot)

            # Choose frame and crop it
            out_frame_dir = out_cam_dir / 'color'
            out_frame_dir.mkdir(parents=True, exist_ok=True)
            out_mask_dir = out_cam_dir / 'mask'
            out_mask_dir.mkdir(parents=True, exist_ok=True)
            out_marker_dir = out_cam_dir / 'gt_marker'
            out_marker_dir.mkdir(parents=True, exist_ok=True)
            
            idx = 0
            for n in range(len(im_paths)):
                if n % pick_frame == 0:
                    fn = im_paths[n].stem

                    im1 = cv2.imread(str(im_paths[n]))
                    kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))
                    mask1 = cv2.imread(str(mask_paths[n]))[:,:,0]

                    kp1[:, 0] -= (left + 1.5)
                    kp1[:, 1] -= (top + 1.5)

                    im1[mask1==0] = 0
                    im1_crop = im1[top:bot, left:right, :]
                    mask1_crop = mask1[top:bot, left:right]

                    cv2.imwrite(str(out_frame_dir / f"{idx:05d}.jpg"), im1_crop)
                    cv2.imwrite(str(out_mask_dir / f"{idx:05d}.png"), mask1_crop)
                    np.savetxt(str(out_marker_dir / f"{idx:05d}.txt"), kp1, delimiter=' ', fmt='%.4f')
                    
                    idx += 1

if thing == "cloth":
    in_dir = Path('D:/raw/cloths')
    out_dir = Path('D:/data/')

    for ID in ID_list:
        out_thing_dir = out_dir / f'{thing}{ID}'
        out_thing_dir.mkdir(parents=True, exist_ok=True)
        for cam in cam_list:
            out_cam_dir = out_thing_dir / f'cam0{cam}'
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            
            # Video to images
            video_dir = in_dir / f'{ID}/ref/cam0{cam}'
            video_dir.mkdir(parents=True, exist_ok=True)
            cmd = f'ffmpeg -i {video_dir}.mp4 {video_dir}/%05d.png'
            print(cmd)
            os.system(cmd)
            os.remove(f'{video_dir}/00001.png')

            # Check images
            im_paths = sorted(list(video_dir.glob("[!.]*.png")))
            kp_dir = in_dir / f'{ID}/markers2d_ref'
            kp_paths = sorted(list(kp_dir.glob(f"[!.]*{cam}.txt")))
            # mask_dir = in_dir / f'{ID}/template_mask'
            # mask_paths = sorted(list(mask_dir.glob(f"[!.]*{cam}.png")))

            # Find bouding box
            # H, W = cv2.imread(str(im_paths[0])).shape[:2]
            # left, right, top, bot = find_box(kp_paths, (H, W), pad=30)
            # print(left, right, top, bot)
            # left, right, top, bot = refine_box(left, right, top, bot, (H, W))
            # print(left, right, top, bot)

            # Choose frame and crop it
            out_frame_dir = out_cam_dir / 'color'
            out_frame_dir.mkdir(parents=True, exist_ok=True)
            # out_mask_dir = out_cam_dir / 'mask'
            # out_mask_dir.mkdir(parents=True, exist_ok=True)
            out_marker_dir = out_cam_dir / 'gt_marker'
            out_marker_dir.mkdir(parents=True, exist_ok=True)
            
            idx = 0
            for n in range(len(im_paths)):
                if n % pick_frame == 0:
                    fn = im_paths[n].stem

                    im1 = cv2.imread(str(im_paths[n]))
                    kp1 = np.float32(np.loadtxt(str(kp_paths[n]), dtype=float))
                    # mask1 = cv2.imread(str(mask_paths[n]))[:,:,0]

                    # kp1[:, 0] -= (left + 1.5)
                    # kp1[:, 1] -= (top + 1.5)

                    # im1[mask1==0] = 0
                    # im1_crop = im1[top:bot, left:right, :]
                    # H, W = im1_crop.shape[:2]
                    # mask1_crop = mask1[top:bot, left:right]

                    cv2.imwrite(str(out_frame_dir / f"{idx:05d}.jpg"), im1)
                    # cv2.imwrite(str(out_mask_dir / f"{idx:05d}.png"), mask1_crop)
                    np.savetxt(str(out_marker_dir / f"{idx:05d}.txt"), kp1, delimiter=' ', fmt='%.4f')
                    
                    idx += 1