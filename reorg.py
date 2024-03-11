from pathlib import Path
import numpy as np
import shutil

in_dir = Path("C:/Users/guoyu/Downloads/1_ref/1/ref")
out_dir = Path("C:/Users/guoyu/Downloads/1_ref/1/raw")
out_dir.mkdir(exist_ok=True)

for cam in np.arange(32, 42):
	cam_folder = out_dir / f'cam0{cam}'
	raw_folder = cam_folder / 'raw'
	cam_folder.mkdir(exist_ok=True)
	raw_folder.mkdir(exist_ok=True)


im_paths = sorted(list(in_dir.glob("[!.]*.png")))

for i, im_path in enumerate(im_paths):
	folder = out_dir / f'cam0{i % 10 + 32}' / 'raw'
	shutil.move(im_path, folder)


in_dir = Path("C:/Users/guoyu/Downloads/1_markers2d_ref/1/markers2d_ref")
out_dir = Path("C:/Users/guoyu/Downloads/1_ref/1/raw")

kp_paths = sorted(list(in_dir.glob("[!.]*.txt")))

for i, kp_path in enumerate(kp_paths):
	folder = out_dir / f'cam0{i % 10 + 32}' / 'raw'
	shutil.move(kp_path, folder)

