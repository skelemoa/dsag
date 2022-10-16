# Usage python3 preprocess_finger_motion.py NTU-X-single-subample.h5
# Requirements: pip install h5py open3d 
# Requirements(optional): pip install smplx open3d 

import os
import sys
import open3d as o3d 
import matplotlib.pyplot as plt
import numpy as np 
import h5py 
from scipy.spatial.transform import Rotation
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm


JOINT_NAMES = [
	'pelvis',
	'left_hip',
	'right_hip',
	'spine1',
	'left_knee',
	'right_knee',
	'spine2',
	'left_ankle',
	'right_ankle',
	'spine3',
	'left_foot',
	'right_foot',
	'neck',
	'left_collar',
	'right_collar',
	'head',
	'left_shoulder',
	'right_shoulder',
	'left_elbow',
	'right_elbow',
	'left_wrist',
	'right_wrist',
	'left_index1',
	'left_index2',
	'left_index3',
	'left_middle1',
	'left_middle2',
	'left_middle3',
	'left_pinky1',
	'left_pinky2',
	'left_pinky3',
	'left_ring1',
	'left_ring2',
	'left_ring3',
	'left_thumb1',
	'left_thumb2',
	'left_thumb3',
	'right_index1',
	'right_index2',
	'right_index3',
	'right_middle1',
	'right_middle2',
	'right_middle3',
	'right_pinky1',
	'right_pinky2',
	'right_pinky3',
	'right_ring1',
	'right_ring2',
	'right_ring3',
	'right_thumb1',
	'right_thumb2',
	'right_thumb3',
	'nose',
	'right_eye',
	'left_eye',
	'right_ear',
	'left_ear',
	'left_big_toe',
	'left_small_toe',
	'left_heel',
	'right_big_toe',
	'right_small_toe',
	'right_heel',
	'left_thumb',
	'left_index',
	'left_middle',
	'left_ring',
	'left_pinky',
	'right_thumb',
	'right_index',
	'right_middle',
	'right_ring',
	'right_pinky',
]



def matrot2axisangle(matrots):
	'''
	:param matrots: N*num_joints*9
	:return: N*num_joints*3
	'''
	import cv2
	batch_size = matrots.shape[0]
	matrots = matrots.reshape([batch_size,-1,9])
	out_axisangle = []
	for mIdx in range(matrots.shape[0]):
		cur_axisangle = []
		for jIdx in range(matrots.shape[1]):
			a = cv2.Rodrigues(matrots[mIdx, jIdx:jIdx + 1, :].reshape(3, 3))[0].reshape((1, 3))
			cur_axisangle.append(a)

		out_axisangle.append(np.array(cur_axisangle).reshape([1,-1,3]))
	return np.vstack(out_axisangle)


def plot_interpolated_values(finger_conf_score,valid_timesteps,valid_rotations,pred_output,sequence_length):
	valid_rotations = matrot2axisangle(valid_rotations.as_matrix()).reshape(-1,3)
	pred_output = matrot2axisangle(pred_output).reshape(-1,3)


	print(valid_rotations.shape)
	print(pred_output.shape)

	import matplotlib.pyplot as plt
	
	timesteps = list(range(len(pred_output)))
	for i in range(3):
		plt.scatter(valid_timesteps,valid_rotations[:,i],label=i)
		plt.plot(timesteps,pred_output[:,i],label=i)

	plt.plot(timesteps,finger_conf_score,label="conf")	
	plt.legend()

	plt.show()	


def visualize_smpl(input_pose,pred_pose,sequence_length,label): 	
		
	assert input_pose.shape[1:] == (52,3,3) and pred_pose.shape[1:] == (52,3,3), f"Wrong input to visualization required Tx52x3x3 got: {input_pose.shape} and {pred_pose.shape}"	

	import torch
	from smplx import SMPLX
	from smplx.utils import Struct

	# smplx does not support .npz by default, so have to load in manually
	smpl_dict = np.load("./smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
	data_struct = Struct(**smpl_dict)
	# print(smpl_dict.files)

	kwargs = {
			'batch_size' : input_pose.shape[0],
			'use_pca' : False,
			'flat_hand_mean' : True
	}


	body_model = SMPLX("./smplx/SMPLX_NEUTRAL.npz",**kwargs).float()

	# Load Updated vertices
	input_pose = torch.from_numpy(matrot2axisangle(input_pose)).view(-1,52*3).float()
	input_smpl_vertices = body_model(body_pose=input_pose[:,1*3:22*3],left_hand_pose=input_pose[:,22*3:37*3],right_hand_pose=input_pose[:,37*3:])
	input_smpl_vertices = input_smpl_vertices.vertices.detach().cpu().numpy()

	pred_pose = torch.from_numpy(matrot2axisangle(pred_pose)).view(-1,52*3).float()
	output_smpl_vertices = body_model(body_pose=pred_pose[:,1*3:22*3],left_hand_pose=pred_pose[:,22*3:37*3],right_hand_pose=pred_pose[:,37*3:])
	output_smpl_vertices = output_smpl_vertices.vertices.detach().cpu().numpy()


	vis = o3d.visualization.Visualizer()
	vis.create_window(width=1280, height=1280,window_name="Preprocessing compare")
	
	verts = body_model().vertices.detach().cpu().numpy().reshape(-1,3)
	faces = body_model.faces_tensor
	rendered_input_mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(verts),
			o3d.utility.Vector3iVector(faces))

	rendered_output_mesh = o3d.geometry.TriangleMesh(
		o3d.utility.Vector3dVector(verts),
		o3d.utility.Vector3iVector(faces))

	vis.add_geometry(rendered_input_mesh)
	vis.add_geometry(rendered_output_mesh)



	ctr = vis.get_view_control()
	# ctr.set_up([0,1,0])
	# ctr.set_lookat([0,0,0])
	# ctr.set_zoom(0.3)

	pause = False
	for t in range(input_pose.shape[0]): 
		print("Rendering:",t,label)

		if t >= sequence_length:
			break

		rendered_input_mesh.vertices = o3d.utility.Vector3dVector(input_smpl_vertices[t])
		rendered_input_mesh.compute_vertex_normals()	
		rendered_input_mesh.translate([-0.33,0,0])

		rendered_output_mesh.vertices = o3d.utility.Vector3dVector(output_smpl_vertices[t])
		rendered_output_mesh.compute_vertex_normals()	
		rendered_output_mesh.translate([+0.33,0,0])

		time.sleep(0.5)

		vis.update_geometry(rendered_input_mesh)
		vis.update_geometry(rendered_output_mesh)

		if pause: 
			vis.run()
			vis.destroy_window()
			vis.close()
			vis = o3d.visualization.Visualizer()
			vis.create_window(width=1280, height=1280,window_name="Preprocessing compare")

			verts = body_model().vertices.detach().cpu().numpy().reshape(-1,3)
			faces = body_model.faces_tensor
			rendered_input_mesh = o3d.geometry.TriangleMesh(
					o3d.utility.Vector3dVector(verts),
					o3d.utility.Vector3iVector(faces))

			rendered_output_mesh = o3d.geometry.TriangleMesh(
				o3d.utility.Vector3dVector(verts),
				o3d.utility.Vector3iVector(faces))

			vis.add_geometry(rendered_input_mesh)
			vis.add_geometry(rendered_output_mesh)

		else:
			vis.poll_events()
			vis.update_renderer()

		vis.capture_screen_image(f"./show_images/{t}.png") # TODO: Returns segfault
	
	os.system(f"ffmpeg -framerate 5 -i ./show_images/\%d.png ./videos/plot_{len(os.listdir('videos'))}_{label}.mp4")	
	os.system("rm -rf ./show_images/*.png")

def process_finger_joints(poses,confidence_scores,length_mask,y,conf_threshold=0.5,sequence_threshold=0.7,visualize=True):
	"""
		Uses rotation spline to interpolate low confidence score vertices 
		Idea: 
			Due to motion blur alot of openpose sequence has issues with predicting hand pose 
			But openpose also gives a confidence score for each hand joint at each timestep 
			Manually set a threshold if the condidence score is below that then the pose is discarded
			If for all hand joints if confidence_score < conf_threshold for 70% of frames. Discard sequence   


		@params: pose (Tx52x3x3) the pose data
		@params: conf_threshold confidence score for all finger joints 
		@params: sequence_threshold to discard sequence   

		@returns: 
			valid_sequences: boolean array mentioning poses which are not discared 
			predicted_pose: pose sequence after interpolating for hand joints 
	"""
	y = np.asarray(y)
	print(y)
	print(length_mask.shape)

	length_mask = np.mean(length_mask,axis=(2,3)) > 0

	sequence_length = length_mask.sum(axis=1)-1

	print(sequence_length)

	y = np.asarray(y)
	pred_pose = poses.copy()

	hand_poses = poses[:,:,22:]
	N,T,J,_,_ = hand_poses.shape

	valid_samples = np.ones((N,T,J),dtype=bool)
	print(valid_samples.shape)

	for n in np.random.choice(N,100):
		# if n%100 == 0:
		# 	print("Starting:",n)
	
		t = sequence_length[n]	
		valid_samples[n,t:] = 0

		for finger_joint in range(J):
			finger_conf_score = confidence_scores[n,:t,finger_joint+22]
			valid_samples[n,:t,finger_joint] = finger_conf_score > conf_threshold



			valid_timesteps = np.where(valid_samples[n,:t,finger_joint])[0]
			valid_rotations = hand_poses[n,valid_timesteps,finger_joint]

			assert valid_timesteps.shape[0] == valid_rotations.shape[0], f"Assert timesteps must be equal to rotations, got:{valid_timesteps.shape} {valid_rotations.shape} "

			if len(valid_timesteps) < 2:
				continue


			valid_rotations = Rotation.from_matrix(valid_rotations)
			spline = RotationSpline(valid_timesteps/t, valid_rotations)
			pred_timesteps = np.arange(0,1 ,1/t)	

			pred_output = spline(pred_timesteps).as_matrix()


			# print("Output:",pred_output.shape)

			pred_pose[n,:t,finger_joint+22] = pred_output 

			if visualize and finger_joint == 14:
				print(n,finger_joint)
				plot_interpolated_values(finger_conf_score,valid_timesteps,valid_rotations,pred_output,t)

		if visualize:

			visualize_smpl(poses[n],pred_pose[n],t,y[n])

			# Convert to rotation matrix 



	valid_samples = np.mean(valid_samples,axis=(1,2)) > sequence_threshold

	return valid_samples,pred_pose

# Questions: 
# Should this also be done on the test set. Or only on the training set. Need to get results first for that

def batch_rodrigues(angle,rot_dir, epsilon=1e-8, dtype=torch.float32):
	''' Calculates the rotation matrices for a batch of rotation vectors
		Parameters
		----------
		rot_vecs: torch.tensor Nx3
			array of N axis-angle vectors
		Returns
		-------
		R: torch.tensor Nx3x3
			The rotation matrices for the given axis-angle parameters
	'''

	batch_size = angle.shape[0]
	device = angle.device

	cos = torch.unsqueeze(torch.cos(angle), dim=1).unsqueeze(dim=2)
	sin = torch.unsqueeze(torch.sin(angle), dim=1).unsqueeze(dim=2)


	# Bx1 arrays
	rx, ry, rz = torch.split(rot_dir, 1, dim=1)
	K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

	zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
	K = torch.cat([zeros.float(), -rz.float(), ry.float(), rz.float(), zeros.float(), -rx.float(), -ry.float(), rx.float(), zeros.float()], dim=1) \
		.view((batch_size, 3, 3))

	ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
	rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
	return rot_mat

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(dim=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)



def cart2polar(vec):
	"""
		Input 
	"""

	XsqPlusYsq = vec[:,0]**2 + vec[:,1]**2
	r = np.sqrt(XsqPlusYsq + vec[:,2]**2)
	theta = np.arctan2(vec[:,2], np.sqrt(XsqPlusYsq))
	phi = np.arctan2(vec[:,1],vec[:,0])

	# Check 
	# print(vec)
	# print(polar2cart(r,theta,phi))
	return r, theta, phi

def polar2cart(r, theta, phi):
	return np.array([
		 r * np.cos(theta) * np.cos(phi),
		 r * np.cos(theta) * np.sin(phi),
		 r * np.sin(theta)]).T



def matrot2axisangle(matrots):
	'''
	:param matrots: N*num_joints*9
	:return: N*num_joints*3
	'''
	import cv2
	batch_size = matrots.shape[0]
	matrots = matrots.reshape([batch_size,-1,9])
	out_axisangle = []
	for mIdx in range(matrots.shape[0]):
		cur_axisangle = []
		for jIdx in range(matrots.shape[1]):
			a = cv2.Rodrigues(matrots[mIdx, jIdx:jIdx + 1, :].reshape(3, 3))[0].reshape((1, 3))
			cur_axisangle.append(a)

		out_axisangle.append(np.array(cur_axisangle).reshape([1,-1,3]))
	return np.vstack(out_axisangle)


def axis_angle_preprocessng(pose_og,mask,label,num_iters=100,learning_rate=10):	
	# For first 75 frame, classes 69-72: lr = 100, lamnda_hand=0.1, ideally body loss: ~2, hand loss: ~10 
	# For every 4th frame, classes 69-72: lr = 10, lamnda_hand=0.1

	# When visually looking at samples: 
	# 1. Look for reduced hand noise (no flickering)
	# 2. For body the wrist should not randomly rotate 


	pose_og = np.array(pose_og)

	mask = np.array(mask)
	sequence_length = np.sum(mask[:,:,0,0],axis=1).astype(np.int32)
	# print("Sequence length:",sequence_length)
	label  = np.array(label)

	N,T,J,_,_ = pose_og.shape

	device = torch.device("cuda")
	# Step 1 Smoothing 
	pose_torch = torch.from_numpy(pose_og)
	pose_torch = Variable(pose_torch,requires_grad=True).to(device)



	for n_iter in range(100): 
		loss_body = (pose_torch[:,1:,:22] - pose_torch[:,:-1,:22])**2
		loss_body = loss_body.sum(dim=(1,2,3,4)).mean()

		loss_hand = (pose_torch[:,1:,22:] - pose_torch[:,:-1,22:])**2
		loss_hand = loss_hand.sum(dim=(1,2,3,4)).mean()


		loss = loss_body + 0.1*loss_hand

		pose_torch.retain_grad()

		loss.backward()

		pose_torch.data -= learning_rate * pose_torch.grad.data


		print(f"Smoothing Iter:{n_iter} Total Loss:{loss.item()} Hand Loss:{loss_hand.item()} Body:{loss_body.item()}")

		pose_torch.grad.data.zero_()

	pose = pose_torch.detach().cpu().numpy()	
	pred_pose = pose.copy()
	# Step 2 Making sure fingers have single axis of rotations
	for finger_joint in range(22,52):
		# Eigen vectors 

		E = [ R for n,T_R in enumerate(pose[:,:,finger_joint]) for tmp_step, R in enumerate(T_R) if tmp_step < sequence_length[n]] # Eigen decomposition data
		E = np.array(E)

		axis_angle_rep = matrot2axisangle(E).reshape(-1,3)

		r,theta,phi = cart2polar(axis_angle_rep)

		median_axis_sperical = np.array([[np.median(theta),np.median(phi)]])
		median_axis_vector = polar2cart(np.ones(1), median_axis_sperical[:,0], median_axis_sperical[:,1])
		# print(median_axis_sperical.shape,median_axis_sperical)

		# plt.scatter(theta,phi,label="Theta+Phi")
		# plt.scatter(median_axis_sperical[:,0],median_axis_sperical[:,1],label="Median")
		# plt.show()
		# Variables wrap a Tensor
		r_torch = Variable(torch.from_numpy(r), requires_grad=True).to(device)
		axis_vector_torch = torch.from_numpy(np.tile(median_axis_vector,(E.shape[0],1))).to(device)

		target_rotation = torch.from_numpy(E).to(device)

		for n_iter in range(num_iters):

			pred_rotation = batch_rodrigues(r_torch,axis_vector_torch)


			rec_loss = (pred_rotation[:,:,:2] - target_rotation[:,:,:2])**2
			rec_loss = rec_loss.mean()


			# smoooth_loss = (pred_rotation[1:,:,:2] - pred_rotation[:-1,:,:2])**2
			# smoooth_loss = smoooth_loss.mean()

			loss = rec_loss

			# print(f"Finger:{finger_joint} Loss:{loss.item()}")
			r_torch.retain_grad()

			loss.backward()

			r_torch.data -= learning_rate * r_torch.grad.data

			r_torch.grad.data.zero_()


		print(f"Finger:{finger_joint} Loss:{loss.item()}")
		pred_rotation = batch_rodrigues(r_torch,axis_vector_torch).detach().cpu().numpy()
		
		cur_ind = 0
		for n in range(N):
			t = sequence_length[n]	
			pred_pose[n,:t,finger_joint] = pred_rotation[cur_ind:cur_ind+t]
			cur_ind += t

		assert cur_ind == E.shape[0], "Error in assigning rotations back to pose"

	# Uncomment to visualize
	# for n in np.random.choice(N,100):
	# 	t = sequence_length[n]
	# 	print(n,label[n])
	# 	visualize_smpl(pose_og[n],pose[n],t,label[n])

	return pred_pose


def getLength(x):
		T = 75
		x = x.reshape((-1 ,T, 52, 3, 3))
		N,J,D,_,_ = x.shape
		data_len = []
		for i in range(x.shape[0]):
			t = T-1
			if np.sum(x[i,T-1,:,:]) == np.sum(x[i,T-2,:,:]):
				for t in range(T-1, 0, -1):
					if np.sum(x[i,t,:,:]) != np.sum(x[i,t-1,:,:]):
						break
			data_len.append(t)
		return np.array(data_len)


def create_mask(data_len):
		T = 75
		data_len = torch.tensor(data_len)
		max_len = data_len.data.max()
		batch_size = data_len.shape[0]
		seq_range = torch.arange(0,T).long()

		seq_range_ex = seq_range.unsqueeze(0).expand(batch_size, T)
		seq_range_ex = seq_range_ex.unsqueeze(2)
		seq_range_ex = seq_range_ex.expand(batch_size, T, 156)

		seq_len = data_len.unsqueeze(1).expand(batch_size, T)
		seq_len = seq_len.unsqueeze(2)
		seq_len = seq_len.expand(batch_size, T, 156)

		return np.array(seq_range_ex < seq_len)


def preprocess_pose(pose, y):
	batch_size = 100
	updated_pose_list = []

	length = getLength(pose)
	mask = create_mask(length)
	mask = mask.reshape((mask.shape[0], mask.shape[1], -1, 3))

	print("Pose Size: ", pose.shape, y.shape)

	# for i in np.unique(y):
	# 	idx = np.where(y == i)
	# 	updated_pose = axis_angle_preprocessng(pose[idx], mask[idx], y[idx])
	# 	print("Output Pose of class index {}: {}".format(i, updated_pose.shape))
	# 	for j in range(updated_pose.shape[0]):
	# 		updated_pose_list.append(updated_pose[j])

	for i in tqdm(range(pose.shape[0]//batch_size + 1)):
		start = i*batch_size
		end = (i+1)*batch_size
		updated_pose = axis_angle_preprocessng(pose[start:end], mask[start:end], y[start:end])
		print("Output Pose: ", updated_pose.shape)
		for j in range(updated_pose.shape[0]):
			updated_pose_list.append(updated_pose[j])
	return np.array(updated_pose_list)


def preprocess_two_persons(pose, y):
	pose1 = pose[:,:,0,:,:,:]
	pose2 = pose[:,:,1,:,:,:]

	pose1 = preprocess_pose(pose1, y)
	pose2 = preprocess_pose(pose2, y)

	pose[:,:,0,:,:,:] = pose1
	pose[:,:,1,:,:,:] = pose2
	return pose


if __name__ == "__main__": 
	datapath = sys.argv[1]
	data = h5py.File(datapath,"r+")
	# process_finger_joints(np.asarray(data['pose']),data['openpose_confidence'],data['mask'],data['y'])

	# Use eigen vectors to find axis of rotation and solve
	updated_train_pose = preprocess_two_persons(data['pose'][:],data['y'][:])
	print("train pose: ", updated_train_pose.shape)
	updated_test_pose = preprocess_two_persons(data['test_pose'][:],data['test_y'][:])
	print("test pose: ", updated_test_pose.shape)

	data.create_dataset("pose_preprocessed",data=updated_train_pose)
	data.create_dataset("test_pose_preprocessed",data=updated_test_pose)
	data.close()


	file = h5py.File(datapath,"r+")
	for k in file: 
		print(k,file[k].shape)