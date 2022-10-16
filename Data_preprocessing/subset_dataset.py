import h5py
import numpy as np
import sys
import os

# a1 = [x for x in range(1, 50)]
# a2 = [x for x in range(61, 106)]

# class_id = a1 + a2
class_id = [69,	70, 71, 72]
# print(len(class_id))
main_path = "/ssd_scratch/cvit/debtanu.gupta/"

leg_cls = [23, 25, 87, 88, 90]

def getData(f, mode="Train", oversample=50):
	if mode == "Train":
		x = f['x'][:][:,:64,:,:]
		pose = f['pose'][:][:,:64,:,:,:]
		mean_pose = f['mean_pose'][:][:,0,:,:]
		y = f['y'][:]
		mask = f['mask'][:]
	else:
		x = f['test_x'][:]
		pose = f['test_pose'][:]
		mean_pose = f['test_mean_pose'][:][:,0,:,:]
		y = f['test_y'][:]
		mask = f['test_mask'][:]

	print(x.shape, pose.shape, mean_pose.shape)

	x_list = []
	pose_list = []
	mean_pose_list = []
	y_list = []
	mask_list = []

	for i, c in enumerate(class_id):
		idx = np.where(c==y)
		x1 = x[idx]
		pose1 = pose[idx]
		mean_pose1 = mean_pose[idx]
		mask1 = mask[idx]
		print(c, pose1.shape)
		for j in range(x1.shape[0]):
			if c in leg_cls:
				for _ in range(oversample):
					x_list.append(x1[j])
					pose_list.append(pose1[j])
					mean_pose_list.append(mean_pose1[j])
					mask_list.append(mask[j])
					y_list.append(i)
			else:	
				x_list.append(x1[j])
				pose_list.append(pose1[j])
				mean_pose_list.append(mean_pose1[j])
				mask_list.append(mask[j])
				y_list.append(i)

	x = np.array(x_list)
	print("3D data: ", x.shape)
	pose = np.array(pose_list)
	mean_pose = np.array(mean_pose_list)
	y = np.array(y_list)
	mask = np.array(mask_list)

	return x, pose, mean_pose, y, mask




if __name__ == "__main__":
	f = h5py.File(os.path.join(main_path, "data", "NTU-X-init-single.h5"), 'r')

	x, pose, mean_pose, y, mask = getData(f, oversample=1)
	print(x.shape, pose.shape, mean_pose.shape, y.shape, mask.shape)

	test_x, test_pose, test_mean_pose, test_y, test_mask = getData(f, mode="Test", oversample=1)
	print(test_x.shape, test_pose.shape, test_mean_pose.shape, test_y.shape, test_mask.shape)

	f = h5py.File(os.path.join(os.path.join(main_path, "data"), "NTU-Xpose-init-4.h5"), 'w')
	f.create_dataset('x', data=x)
	f.create_dataset('pose', data=pose)
	f.create_dataset('mean_pose', data=mean_pose)
	f.create_dataset('y', data=y)
	f.create_dataset('mask', data=mask)

	f.create_dataset('test_x', data=test_x)
	f.create_dataset('test_pose', data=test_pose)
	f.create_dataset('test_mean_pose', data=test_mean_pose)
	f.create_dataset('test_y', data=test_y)
	f.create_dataset('test_mask', data=test_mask)
	f.close()
