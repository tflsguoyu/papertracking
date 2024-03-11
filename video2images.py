import os
import numpy as np
os.system(f'ffmpeg -i D:/raw/ropes/1/ref/cam032.mp4 D:/data/rope1/cam032/color/frame%06d_cam032_normal.png')

# for cam in np.arange(32, 42):
# 	os.system(f'ffmpeg -i C:/Users/guoyu/Downloads/1_ref/1/ref/cam0{cam}.mp4 C:/Users/guoyu/Downloads/1_ref/1/ref/frame%06d_cam0{cam}.png')


# for cam in np.arange(32, 42):
# 	os.system(f'ffmpeg -i C:/Users/guoyu/Downloads/4_video/ref/cam0{cam}.mp4 C:/Users/guoyu/Downloads/4_video/ref/frame%06d_cam0{cam}.png')

# for cam in np.arange(32, 42):
# 	os.system(f'ffmpeg -i C:/Users/guoyu/Downloads/7_video/ref/cam0{cam}.mp4 C:/Users/guoyu/Downloads/7_video/ref/frame%06d_cam0{cam}.png')
