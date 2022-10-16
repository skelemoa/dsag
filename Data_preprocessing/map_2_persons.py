import numpy as np
import h5py
import sys
import os
from tqdm import tqdm


main_path_xpose = "./Dataset/data"
main_path_kinect = './Dataset/kinect/'
kinect_files = os.listdir(main_path_kinect)


def get_vibe_dir(x):
    x1 = x[16,:] - x[0,:]
    x2 = x[17,:] - x[0,:]
    return np.cross(x1,x2)


def get_kinect_dir(x):
    x1 = x[8,:] - x[0,:]
    x2 = x[4,:] - x[0,:]
    return np.cross(x1, x2)

def get_kinect_peron(i):
    # this function returns Kinect skeleton given the index
    f = i + '.skeleton.npy'
    p = np.zeros((300,2,25,3))
    if f not in kinect_files:
#         print(f)
        pass
    else:
        kp = np.load(os.path.join(main_path_kinect, f))
        if kp.shape[0] != 0:
            if kp.shape[0] == 1:
                p[:,0,:,:] = kp[0,:,:,:]
            else:
                p[:,0,:,:] = kp[0,:,:,:]
                p[:,1,:,:] = kp[1,:,:,:]
    return p[::4,:,:,:]


def order_root(kinect_person, vibe, mean_pose, pose):
    vibe = vibe.reshape((2, 52, 3))
    
    person = vibe[:,:,:]
    root_left = kinect_person[0,0,:].reshape((1,3))
    root_right = kinect_person[1,0,:].reshape((1,3))

#     person1 = person[0,:,:] + left
#     person2 = person[1,:,:] + right
    left = person[0,:,:]
    right = person[1,:,:]

    left_mean_pose = mean_pose[0,:,:]
    right_mean_pose = mean_pose[1,:,:]

    left_pose = pose[0,:,:,:]
    right_pose = pose[1,:,:,:]

    v1 = get_vibe_dir(person[0,:,:])
    v2 = get_vibe_dir(person[1,:,:])
    v_cross = np.cross(v1, v2)

    k1 = get_kinect_dir(kinect_person[0,:,:])
    k2 = get_kinect_dir(kinect_person[1,:,:])
    k_cross = np.cross(k1,k2)
    
    dot_prod = np.sum(v_cross*k_cross)
#     print(dot_prod)

    if dot_prod > 0:
        # right direction
        return left, right, root_left, root_right, left_mean_pose, right_mean_pose, left_pose, right_pose
    elif dot_prod < 0:
        # Wrong Direction
        return right, left, root_left, root_right, right_mean_pose, left_mean_pose, right_pose, left_pose
    else:
        # one person missing
        return left, right, root_left, root_right, left_mean_pose, right_mean_pose, left_pose, right_pose


def getDataset(x, y, file, mean_pose, pose):
	x_list = []
	root_list = []
	mean_pose_list = []
	pose_list = []
	y_list = []

	for i in tqdm(range(x.shape[0])):
		kinect = get_kinect_peron(file[i].astype(str))
		xpose = x[i]
		xpose_mean_pose = mean_pose[i]
		xpose_pose = pose[i]

		sequence = np.zeros(xpose.shape)
		root = np.zeros((xpose.shape[0], 2, 1, 3))
		new_mean_pose = np.zeros(xpose_mean_pose.shape)
		new_pose = np.zeros(xpose_pose.shape)

		for j in range(xpose.shape[0]):
			if np.sum(xpose[j]) == 0:
				break
			p1, p2, root_left, root_right, m1, m2, pose1, pose2 = order_root(kinect[j], xpose[j], xpose_mean_pose[j], xpose_pose[j])
			sequence[j,0,:,:] = p1
			sequence[j,1,:,:] = p2
			root[j,0,:,:] = root_left
			root[j,1,:,:] = root_right

			new_mean_pose[j,0,:,:] = m1
			new_mean_pose[j,1,:,:] = m2

			new_pose[j,0,:,:,:] = pose1
			new_pose[j,1,:,:,:] = pose2

		x_list.append(sequence)
		root_list.append(root)
		mean_pose_list.append(new_mean_pose)
		pose_list.append(new_pose)
		y_list.append(y[i])

	return np.array(x_list), np.array(root_list), np.array(mean_pose_list), np.array(pose_list), np.array(y_list)



if __name__ == "__main__":
	f = h5py.File(os.path.join(main_path_xpose, "NTU-X-Multi-Person.h5"), 'r')
	x = f['x'][:]
	y = f['y'][:]
	file = f['file_name'][:]
	mean_pose = f['mean_pose'][:]
	pose = f['pose'][:]
	mask = f['mask'][:]

	test_x = f['test_x'][:]
	test_y = f['test_y'][:]
	test_file = f['test_file_name'][:]
	test_mean_pose = f['test_mean_pose'][:]
	test_pose = f['test_pose'][:]
	test_mask = f['test_mask'][:]
	
	print(x.shape, y.shape, file.shape, mean_pose.shape, pose.shape)
	print(test_x.shape, test_y.shape, test_file.shape, test_mean_pose.shape, test_pose.shape)

	x, root, mean_pose, pose, y = getDataset(x, y, file, mean_pose, pose)
	print(x.shape, root.shape, mean_pose.shape, pose.shape, y.shape)
	
	test_x, test_root, test_mean_pose, test_pose, test_y = getDataset(test_x, test_y, test_file, test_mean_pose, test_pose)
	print(test_x.shape, test_root.shape, test_mean_pose.shape, test_pose.shape, test_y.shape)

	f = h5py.File(os.path.join(main_path_xpose, "NTU-X-Global-2-Persons.h5"), 'w')
	f.create_dataset("x", data=x)
	f.create_dataset("root", data=root)
	f.create_dataset("mean_pose", data=mean_pose)
	f.create_dataset("pose", data=pose)
	f.create_dataset("y", data=y)
	f.create_dataset("mask", data=mask)
	f.create_dataset("file_name", data=file)

	f.create_dataset("test_x", data=test_x)
	f.create_dataset("test_root", data=test_root)
	f.create_dataset("test_mean_pose", data=test_mean_pose)
	f.create_dataset("test_pose", data=test_pose)
	f.create_dataset("test_y", data=test_y)
	f.create_dataset("test_mask", data=test_mask)
	f.create_dataset("test_file_name", data=test_file)
	f.close()

	




		