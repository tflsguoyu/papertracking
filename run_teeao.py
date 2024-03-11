import os 

thing = "rope"
ID_list = [14,15,19]
cam_list = [32]
for ID in ID_list:
	for cam in cam_list:
		os.chdir('C:/Users/guoyu/cvpr/omnimotion/preprocessing/')
		os.system(f'python main_processing.py --data_dir D:/data/{thing}{ID}/cam0{cam} --chain')

		os.chdir('C:/Users/guoyu/cvpr/omnimotion/')
		os.system(f'python train.py --config configs/default.txt --expname {thing}{ID} --data_dir D:/data/{thing}{ID}/cam0{cam}')

		# exit()
		# os.chdir('C:/Users/guoyu/cvpr/omnimotion/')
		# os.system(f'python viz.py --config configs/default.txt --expname rope{ropeID} --data_dir D:/data/rope{ropeID}/cam0{cam}')


# ropeID_list = [1]
# for ropeID in ropeID_list:

# 	cam_list = [32]
# 	for cam in cam_list:
# 		os.chdir('C:/Users/guoyu/cvpr/omnimotion/preprocessing/')
# 		os.system(f'python main_processing.py --data_dir C:/Users/guoyu/data/rope{ropeID}/cam0{cam} --chain')

# 		os.chdir('C:/Users/guoyu/cvpr/omnimotion/')
# 		os.system(f'python train.py --config configs/default.txt --expname rope{ropeID} --data_dir C:/Users/guoyu/data/rope{ropeID}/cam0{cam}')

# 		os.chdir('C:/Users/guoyu/cvpr/omnimotion/')
# 		os.system(f'python viz.py --config configs/default.txt --expname rope{ropeID} --data_dir C:/Users/guoyu/data/rope{ropeID}/cam0{cam}')