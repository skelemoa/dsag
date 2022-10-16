import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
from rotation import rot6d_to_rotmat, batch_rigid_transform
import h5py
import numpy as np
import os
from tqdm import tqdm
import sys


main_path = "./Dataset"
SMPL = np.load("SMPLX_NEUTRAL.npz")
# print(list(SMPL.keys()))
J_regressor = SMPL['J_regressor']
parents = SMPL['kintree_table'][0]
parents[0] = -1
select_classes = [1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
                    27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  61,  62,  63,
                    64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
                    77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                    90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
                   103, 104, 105]


def getData(global_orient, body_pose, left_hand_pose, right_hand_pose, betas):
	mean_pose = SMPL['v_template'][:,:,None] + SMPL['shapedirs'][:,:,:10]@betas.T
	mean_pose = mean_pose.transpose((1,0,2))
	mean_pose = SMPL['J_regressor']@mean_pose
	mean_pose = mean_pose.transpose((2,1,0))

	t = global_orient.shape[0]

	# extract left and right hand from mean pose
	left = mean_pose[:,25:25+15]
	right = mean_pose[:,25+15:]
	left_parent = np.array(np.array(parents[25:25+15])) - 25
	left_parent[np.where(left_parent == -5)] = 0

	right_parent = np.array(np.array(parents[25+15:])) - 40
	right_parent[np.where(right_parent == -19)] = 0

	# create new t-pose
	t_pose = np.zeros((t,52,3))
	t_pose[:,:22,:] = mean_pose[:,:22,:]
	t_pose[:,22:22+15,:] = left
	t_pose[:,22+15:,] = right

	# create modified parent array
	left_parent1 = left_parent + 22
	left_parent1[np.where(left_parent1==22)] = 20

	right_parent1 = right_parent + 22 + 15
	right_parent1[np.where(right_parent1==37)] = 21
	parent_array = list(parents[:22]) + list(left_parent1) + list(right_parent1)
	parent_array = np.array(parent_array)
	
	# forward kinematics
	full_pose = torch.cat([global_orient, body_pose, left_hand_pose,right_hand_pose], dim=1)
	X_transformed = batch_rigid_transform(full_pose.float(), torch.tensor(t_pose).float(), parent_array)
	
	X_transformed = np.array(X_transformed)
	full_pose = np.array(full_pose)

	return X_transformed, full_pose, t_pose



def get_class_index(y):
	y_list = [select_classes.index(x) for x in y]
	return np.array(y_list)


if __name__ == "__main__":
	path = os.path.join(main_path, "smplx")

	x_list = []
	pose_list = []
	t_pose_list = []
	y_list = []
	mask_list = []
	setup_list = []
	file_name_list = []

	for folder1 in os.listdir(path):
		path1 = os.path.join(path, folder1)
		# if folder1 != "NTU_1_49":
		# 	break
		for folder2 in os.listdir(path1):
			path2 = os.path.join(path1, folder2)
			for h5_files in os.listdir(path2):
				print(path2, h5_files)
				try:
					l = h5py.File(os.path.join(path2, h5_files), 'r')
					global_orient = torch.tensor(l['global_orient'][:])
					body_pose = torch.tensor(l['body_pose'][:])
					left_hand_pose = torch.tensor(l['left_hand_pose'][:])
					right_hand_pose = torch.tensor(l['right_hand_pose'][:])
					betas = l['betas'][:]
					file_names = l['file_names'][:]
					l.close()
				except:
					print("{} -> empty file".format(os.path.join(path2, h5_files)))
					continue

				print(global_orient.shape, body_pose.shape, left_hand_pose.shape, right_hand_pose.shape, betas.shape, file_names.shape)

				sample_names = np.array([str(x).split('.')[0].split("\'")[1] for x in file_names])
				frame_id = np.array([int(x.split('_')[-1]) for x in sample_names])
				sample = np.array([x.split('_')[0] for x in sample_names])
				
				for i in tqdm(np.unique(sample)):
					y = int(i[-3:])

					if y not in select_classes:
						continue

					camera = int(i.split("C")[1][:3])
					R = int(i.split("R")[1][:3])
					setup_id = int(i[1:4])

					if (camera == 3 and R == 1) or (camera == 2 and R == 2): # front facing samples for single person classes
						idx = np.where(i == sample)
						sample_frame = frame_id[idx]
						sample_global_orient = global_orient[idx]
						sample_body_pose = body_pose[idx]
						sample_left_hand_pose = left_hand_pose
						sample_right_hand_pose = right_hand_pose[idx]
						sample_betas = betas[idx]
						# t = sample_frame.shape[0]


						sorted_index = np.argsort(sample_frame)
						sample_global_orient = sample_global_orient[sorted_index]
						sample_body_pose = sample_body_pose[sorted_index]
						sample_left_hand_pose = sample_left_hand_pose[sorted_index]
						sample_right_hand_pose = sample_right_hand_pose[sorted_index]
						sample_betas = sample_betas[sorted_index]


						# get person 1 sequence
						x1, rotmat1, mean_pose1 = getData(sample_global_orient[:,0:1,:,:], sample_body_pose[:,0,:,:,:]
												, sample_left_hand_pose[:,0,:,:,:], sample_right_hand_pose[:,0,:,:,:], sample_betas[:,0,:])

						x1 = x1[:75,:,:]
						rotmat1 = rotmat1[:75,:,:]
						mean_pose1 = mean_pose1[:75,:,:]

						# get person 2 sequence
						x2, rotmat2, mean_pose2 = getData(sample_global_orient[:,1:,:,:], sample_body_pose[:,1,:,:,:]
												, sample_left_hand_pose[:,1,:,:,:], sample_right_hand_pose[:,1,:,:,:], sample_betas[:,1,:])

						x2 = x2[:75,:,:]
						rotmat2 = rotmat2[:75,:,:]
						mean_pose2 = mean_pose2[:75,:,:]
						t = x1.shape[0]

						X_transformed = np.zeros((75, 2, 52, 3))
						pose = np.zeros((75, 2, 52, 3, 3))
						t_pose = np.zeros((75, 2, 52, 3))
						mask = np.zeros((75, 2, 52, 3))

						X_transformed[:t,0,:,:] = x1
						pose[:t,0,:,:,:] = rotmat1
						t_pose[:t,0,:,:] = mean_pose1

						X_transformed[:t,1,:,:] = x2
						pose[:t,1,:,:,:] = rotmat2
						t_pose[:t,1,:,:] = mean_pose2

						# X_transformed = X_transformed[::4,:,:]
						# pose = pose[::4,:,:,:]
						# t_pose = t_pose[::4,:,:]
						mask = mask[:75,:,:]

						x_list.append(X_transformed)
						pose_list.append(pose)
						t_pose_list.append(t_pose)
						y_list.append(y)
						setup_list.append(setup_id)
						file_name_list.append(i)
						mask_list.append(mask)
					else:
						continue
						

	x = np.array(x_list)
	pose = np.array(pose_list)
	t_pose = np.array(t_pose_list)
	y = np.array(y_list)
	setup = np.array(setup_list)
	file_name = np.array(file_name_list, dtype='S')
	mask = np.array(mask_list)
	y = get_class_index(y)
	print(x.shape, pose.shape, t_pose.shape, y.shape, file_name.shape, mask.shape)

	train_idx = np.where(setup%2 == 0)
	test_idx = np.where(setup%2 != 0)

	f = h5py.File(os.path.join(os.path.join(main_path, "data"), "NTU-X-Multi-Person.h5"), 'w')
	f.create_dataset('x', data=x[train_idx])
	f.create_dataset('pose', data=pose[train_idx])
	f.create_dataset('mean_pose', data=t_pose[train_idx])
	f.create_dataset('y', data=y[train_idx])
	f.create_dataset('file_name', data=file_name[train_idx])
	f.create_dataset('mask', data=mask[train_idx])

	f.create_dataset('test_x', data=x[test_idx])
	f.create_dataset('test_pose', data=pose[test_idx])
	f.create_dataset('test_mean_pose', data=t_pose[test_idx])
	f.create_dataset('test_y', data=y[test_idx])
	f.create_dataset('test_file_name', data=file_name[test_idx])
	f.create_dataset('test_mask', data=mask[test_idx])
	f.close()
