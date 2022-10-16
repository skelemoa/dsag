import os
import sys
import h5py
import numpy as np
import traceback  
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
# from modelKL import *
from model import *
import math
from rotation import rot6d_to_rotmat, batch_rigid_transform, rotmat_to_rot6d
from mmd import mmd_function


class_name = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up", "clapping"
			, "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe", "take off a shoe", "put on glasses"
			, "take off glasses", "put on a hat_cap", "take off a hat_cap", "cheer up", "hand waving", "kicking something", "reach into pocket"
			, "hopping", "jump up", "phone call", "play with phone_tablet", "type on a keyboard", "point to something", "taking a selfie"
			, "check time (from watch)", "rub two hands", "nod head_bow", "shake head", "wipe face", "salute", "put palms together"
			, "cross hands in front", "sneeze_cough", "staggering", "falling down", "headache", "chest pain", "back pain", "neck pain"
			, "nausea_vomiting", "fan self", "punch_slap", "kicking", "pushing", "pat on back", "point finger", "hugging", "giving object"
			, "touch pocket", "shaking hands", "walking towards", "walking apart", "put on headphone", "take off headphone", "shoot at basket"
			, "bounce ball", "tennis bat swing", "juggle table tennis ball", "hush", "flick hair", "thumb up", "thumb down", "make OK sign"
			, "make victory sign", "staple book", "counting money", "cutting nails", "cutting paper", "snap fingers", "open bottle", "sniff_smell"
			, "squat down", "toss a coin", "fold paper", "ball up paper", "play magic cube", "apply cream on face", "apply cream on hand", "put on bag"
			, "take off bag", "put object into bag", "take object out of  bag", "open a box", "move heavy objects", "shake fist", "throw up cap_hat"
			, "capitulate", "cross arms", "arm circles", "arm swings", "run on the spot", "butt kicks", "cross toe touch", "side kick", "yawn"
			, "stretch oneself", "blow nose", "hit with object", "wield knife", "knock over", "grab stuff", "shoot with gun", "step on foot"
			, "high-five", "cheers and drink", "carry object", "take a photo", "follow", "whisper", "exchange things", "support somebody", "rock-paper-scissors"]

print("Number of classes: {}".format(len(class_name)))
parent_array=np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 20, 23, 20
                         , 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 21, 38, 21, 40, 41, 21, 43, 44, 21, 46
                         , 47, 21, 49, 50])

leg_cls = [23, 25, 87, 88, 90]
person_2_cls = [x-1 for x in range(50, 61)] + [x-1 for x in range(106, 120)]

main_path = "./Dataset"
learning_rate = 1.5e-2
batch_size = 100
max_epochs = 400
latent_dim = 768
num_workers = 10
device = torch.device('cuda')
load_epoch = -1
num_class = 120
lambda1 = 10
lambda2 = 0.1
lambda_kld = 0.1

class NTUDataset(Dataset):
	def __init__(self, data):
		# print(x.shape, rot6d.shape, mask.shape, y.shape, mean_pose.shape)
		self.x = data[0][:,:,:]
		self.rot6d = data[1][:,:,:,:,:]
		self.mean_pose = data[2] # We keep the skeleton fixed across sequence
		self.N = self.x.shape[0]
		self.label = data[3]
		self.label = self.label - 1
		self.y = np.zeros((self.N,num_class))
		self.y[np.arange(self.N),self.label] = 1

		self.length = self.getLength(self.x)
		self.mask = self.create_mask(self.length)

		self.seq = self.get_variable_seq_rep(self.mask)
		
		self.rot6d = rotmat_to_rot6d(self.rot6d)

		print(self.x.shape, self.rot6d.shape, self.mean_pose.shape, self.mask.shape, self.y.shape, self.seq.shape)

		self.root = data[4]
		self.root = self.root.reshape((self.N, 64, 2, -1))
		self.res1 = self.root[:,:,1,:] - self.root[:,:,0,:]
		self.res2 = self.root[:,:,0,:] - np.zeros((self.root[:,:,0,:].shape))

		self.residual = np.zeros((self.root.shape[0], self.root.shape[1], 2*3))
		self.residual[:,:,:3] = self.res2
		self.residual[:,:,3:] = self.res1
		print('Residual shape:', self.res1.shape, self.res2.shape, self.residual.shape)

		self.wrist = np.zeros((self.N, self.x.shape[1], 2, 2, 3))
		self.wrist[:,:,:,0,:] = self.x[:,:,:,21,:]
		self.wrist[:,:,:,1,:] = self.x[:,:,:,20,:]

		self.length_root = self.getLengthMask(self.root)
		self.mask_root = self.create_mask(self.length_root)[:,:,:6]

		self.x = self.x[:,:64,:,:,:]
		self.hand_rot6d = self.rot6d[:,:64,:,22:]
		self.rot6d = self.rot6d[:,:64,:,:22]
		self.mask = self.mask[:,:64,:]
		self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], -1)
		self.rot6d = self.rot6d.reshape(self.rot6d.shape[0], self.rot6d.shape[1], -1)
		self.hand_rot6d = self.hand_rot6d.reshape(self.hand_rot6d.shape[0], self.hand_rot6d.shape[1], -1)
		self.mask = self.mask.reshape(self.mask.shape[0], self.mask.shape[1], -1)
		self.wrist = self.wrist.reshape((self.N, self.wrist.shape[1], -1))

		print(self.x.shape, self.rot6d.shape, self.hand_rot6d.shape, self.y.shape, self.mask.shape, self.mean_pose.shape, self.residual.shape, self.seq.shape, self.mask_root.shape, self.wrist.shape)

	def __len__(self):
		return self.N

	def __getitem__(self, index):
		# if self.length[index] <= max_length:
		# 	T = 0
		# else: 
		# 	T = np.random.choice(self.length[index]-max_length)
		# # return [self.rot6d[index],self.x[index], self.mask[index], self.y[index], self.residual[index], self.mean_pose[index]]
		return self.x[index], self.rot6d[index], self.hand_rot6d[index], self.mean_pose[index], self.mask[index], self.y[index], self.residual[index], self.seq[index], self.mask_root[index], self.wrist[index]

	def getLength(self, x):
		T = 64
		x = x.reshape((-1 ,T, 2, 52, 3))
		N,J,P,D,_ = x.shape
		data_len = []
		for i in range(x.shape[0]):
			t = T-1
			if np.sum(x[i,T-1,:,:,:]) == np.sum(x[i,T-2,:,:,:]):
				for t in range(T-1, 0, -1):
					if np.sum(x[i,t,:,:,:]) != np.sum(x[i,t-1,:,:,:]):
						break
			data_len.append(t)
		return np.array(data_len)

	def getLengthMask(self, x):
		T = 64
		x = x.reshape((-1 ,T, 2, 1, 3))
		N,J,_,_,_ = x.shape
		data_len = []
		for i in range(x.shape[0]):
			t = T-1
			if np.sum(x[i,T-1]) == np.sum(x[i,T-2]):
				for t in range(T-1, 0, -1):
					if np.sum(x[i,t]) != np.sum(x[i,t-1]):
						break
			data_len.append(t)
		return np.array(data_len)

	
	def get_samples(self,labels):
		samples = []
		for l in labels:
			il = np.random.choice(np.where(self.y[:,l]>0)[0])	
			samples.append(self.__getitem__(il)[0])
		return np.array(samples)


	def get_variable_seq_rep(self, mask):
		N,T,_ = mask.shape
		mask = mask.reshape((N,T,2,52,3))
		mask = mask[:,:,0,:,:]
		seq_len = np.sum(mask[:,:,0,0], axis=1)
		idx1 = np.where(seq_len == 64)
		seq_len = seq_len[:, None]
		seq = np.arange(0, 64)
		seq = seq.reshape((1, 64))
		seq = np.repeat(seq, mask.shape[0], axis=0)
		seq = seq/seq_len
		idx = np.where(seq>=1)
		seq[idx] = 1.0
		seq[:,-1] = 1.0
		seq = seq[:,:,None]
		return seq


	def create_mask(self, data_len):
		T = 64
		data_len = torch.tensor(data_len)
		max_len = data_len.data.max()
		batch_size = data_len.shape[0]
		seq_range = torch.arange(0,T).long()

		seq_range_ex = seq_range.unsqueeze(0).expand(batch_size, T)
		seq_range_ex = seq_range_ex.unsqueeze(2)
		seq_range_ex = seq_range_ex.expand(batch_size, T, 156*2)

		seq_len = data_len.unsqueeze(1).expand(batch_size, T)
		seq_len = seq_len.unsqueeze(2)
		seq_len = seq_len.expand(batch_size, T, 156*2)

		return np.array(seq_range_ex < seq_len)



def fkt(x, mean_pose, device, parent_array):
	# forward kinematics
	rotmat = rot6d_to_rotmat(x)
	# same mean pose across timesteps
	mean_pose = mean_pose.reshape((x.shape[0], 1, -1))
	mean_pose = mean_pose.expand((x.shape[0], x.shape[1], 156))
	mean_pose = mean_pose[:,:,:].reshape((x.shape[0]*x.shape[1],-1,3))
	rotmat = rotmat.reshape((x.shape[0]*x.shape[1],-1, 3, 3))
	pred = batch_rigid_transform(rotmat.float(),mean_pose.to(device).float(),parent_array)
	x = pred.reshape((x.shape[0], x.shape[1], 1, 52, 3))
	return x


def loss_function(x, pred_3d, rot6d, hand_rot6d, pred_hand, pred_6d, mask, leg_mask, root, root_pred, mask_root):
	leg = [1,2,4,5,7,8,10,11]

	mask = mask.reshape((mask.shape[0], 64, 2, 52, 3))
	x = x.reshape((x.shape[0], 64, 2, 52, 3))
	maeloss_3d = torch.sum(torch.mul(torch.abs(pred_3d[:,:,:,:22] - x[:,:,:,:22].to(device)), mask[:,:,:,:22].to(device).float())) / (torch.sum(mask[:,:,:,:22].to(device).float()) + 10e-8)
	maeloss_3d_hand = torch.sum(torch.mul(torch.abs(pred_3d[:,:,:,22:] - x[:,:,:,22:].to(device)), mask[:,:,:,22:].to(device).float())) / (torch.sum(mask[:,:,:,22:].to(device).float()) + 10e-8)


	mask1 = mask.unsqueeze(5).expand(mask.shape[0], mask.shape[1], 2, 52, 3, 2).reshape((mask.shape[0], mask.shape[1], -1))
	pred_6d = pred_6d.reshape((pred_6d.shape[0], 64, -1))
	pred_hand = pred_hand.reshape((pred_hand.shape[0], 64, -1))

	maeloss_6d = torch.sum(torch.mul(torch.abs(pred_6d[:,:,:] - rot6d[:,:,:].to(device)), mask1[:,:,:264].to(device).float())) / (torch.sum(mask1[:,:,:264].to(device).float()) + 10e-8)
	maeloss_6d_hand = torch.sum(torch.mul(torch.abs(pred_hand[:,:,:] - hand_rot6d[:,:,:].to(device)), mask1[:,:,264:].to(device).float())) / (torch.sum(mask1[:,:,264:].to(device).float()) + 10e-8)

	rot6d = rot6d.reshape((rot6d.shape[0], rot6d.shape[1], 2, 22, 6))
	pred_6d = pred_6d.reshape((pred_6d.shape[0], pred_6d.shape[1], 2, 22, 6))
	mask1 = mask1.reshape((mask1.shape[0], mask1.shape[1], 2, 52, 6))
	mask1 = mask1[:,:,:,:22,:]

	# leg_6d = torch.sum(torch.mul(torch.abs(pred_6d[:,:,leg,:] - rot6d[:,:,leg, :].to(device)), mask1[:,:,leg,:].to(device).float()))/ (torch.sum(mask1[:,:,leg,:].to(device).float()) + 10e-8)
	l = torch.sum(torch.mul(torch.abs(pred_6d[:,:,:,leg,:] - rot6d[:,:,:,leg, :].to(device)), mask1[:,:,:,leg,:].to(device).float()), dim=(1,2,3,4))
	leg_6d = l / (torch.sum(mask1[:,:,:,leg,:].to(device).float(), dim=(1,2,3,4)) + 10e-8)
	leg_6d = torch.mean(leg_6d*leg_mask)

	# root_mask = mask1[:,:,0,0,:6]
	root_loss = torch.sum(torch.mul(torch.abs(root.float() - root_pred), mask_root.to(device).float()))/(torch.sum(mask_root.to(device).float()) + 10e-8)
	
	return maeloss_3d, maeloss_6d, leg_6d, maeloss_3d_hand, maeloss_6d_hand, root_loss


def train(epoch, model, train_loader, optimizer, L1Loss, L):
	total_loss = 0
	loss_3d = 0
	loss_6d = 0
	loss_kld = 0
	loss_root = 0
	loss_seq = 0

	for i, (x, rot6d, hand_rot6d, mean_pose, mask, y, root, seq, mask_root, wrist) in enumerate(train_loader):
		# lambda_kld = L[i]
		# print(lambda_kld)
		label = torch.argmax(y, dim=1)
		leg_mask = torch.zeros(label.shape)
		for idx, p in enumerate(label):
			if p in leg_cls:
				leg_mask[idx] = 1

		optimizer.zero_grad()
		rot = rot6d[:,0,:].reshape((rot6d.shape[0], 44, 6))
		rot = rot[:,0,:]

		pred, pred_hand, kld, pred_root, seq_pred = model(rot6d.to(device).float(), hand_rot6d.to(device).float(), y.to(device).float(), rot.to(device).float(), root.to(device).float(), seq.to(device).float(), wrist.to(device).float())
		full_body_pred = torch.cat((pred, pred_hand), dim=3)

		pred_3d1 = fkt(full_body_pred[:,:,0,:,:].contiguous(), mean_pose[:,0,:,:].contiguous(), device, parent_array)
		pred_3d2 = fkt(full_body_pred[:,:,1,:,:].contiguous(), mean_pose[:,1,:,:].contiguous(), device, parent_array)

		pred_3d = torch.cat((pred_3d1, pred_3d2), dim=2)
		
		maeloss_3d, maeloss_6d, leg_6d, maeloss_3d_hand, maeloss_6d_hand, root_loss = loss_function(x.to(device).float(), pred_3d
						, rot6d.to(device).float(), hand_rot6d.to(device).float(), pred_hand, pred, mask, leg_mask.to(device), root.to(device).float(), pred_root, mask_root.to(device).float())

		seq_loss = L1Loss(seq.to(device).float(), seq_pred)
		loss = 10*(maeloss_3d + lambda1*maeloss_6d + 10*leg_6d + maeloss_3d_hand + maeloss_6d_hand) + lambda_kld*kld + root_loss + 2*seq_loss
		
		# if i == 0:
		# 	for p in range(label.shape[0]):
		# 		if label[p] in person_2_cls:
		# 			print("Orig: ", root[p, 2])
		# 			print("pred: ", pred_root[p, 2])
		# 			print("=====================================")

		loss.backward()
		optimizer.step()

		total_loss += loss.cpu().data.numpy()*x.shape[0]
		loss_3d += maeloss_3d.cpu().data.numpy()*x.shape[0]
		loss_6d += maeloss_6d.cpu().data.numpy()*x.shape[0]
		loss_kld += kld.cpu().data.numpy()*x.shape[0]
		loss_root += root_loss.cpu().data.numpy()*x.shape[0]
		loss_seq += seq_loss.cpu().data.numpy()*x.shape[0]

	total_loss /= len(train_loader.dataset)
	loss_3d /= len(train_loader.dataset)
	loss_6d /= len(train_loader.dataset)
	loss_kld /= len(train_loader.dataset)
	loss_root /= len(train_loader.dataset)
	loss_seq /= len(train_loader.dataset)

	return total_loss, loss_3d, loss_6d, loss_kld, loss_root, loss_seq




def test(epoch, model, val_loader, L1Loss,plot_samples=0):

	total_loss = 0
	loss_3d = 0
	loss_6d = 0
	loss_kld = 0
	loss_root = 0

	for i,(x, rot6d, hand_rot6d, mean_pose, mask, y, root) in enumerate(val_loader):
		label = torch.argmax(y, dim=1)
		leg_mask = torch.zeros(label.shape)
		for idx, p in enumerate(label):
			if p in leg_cls:
				leg_mask[idx] = 1

		pred, pred_hand, kld = model(rot6d.to(device).float(), hand_rot6d.to(device).float(), y.to(device).float(), root.to(device).float())
		
		full_body_pred = torch.cat((pred, pred_hand), dim=2)
		pred_3d = fkt(full_body_pred, mean_pose, device, parent_array)
		
		maeloss_3d, maeloss_6d, leg_6d, maeloss_3d_hand, maeloss_6d_hand = loss_function(x.to(device).float(), pred_3d, rot6d.to(device).float(), hand_rot6d.to(device).float(), pred_hand, pred, mask, leg_mask.to(device))
		loss = 10*(maeloss_3d + lambda1*maeloss_6d + 10*leg_6d + maeloss_3d_hand + maeloss_6d_hand) + lambda_kld*kld
		
		total_loss += loss.cpu().data.numpy()*x.shape[0]
		loss_3d += maeloss_3d.cpu().data.numpy()*x.shape[0]
		loss_6d += maeloss_6d.cpu().data.numpy()*x.shape[0]
		loss_kld += kld.cpu().data.numpy()*x.shape[0]

		# if i == 0 and epoch %30 == 0:
		# 	plot(epoch, x.cpu().data.numpy(), pred_3d.cpu().data.numpy(), y.cpu().data.numpy(),plot_samples=plot_samples)

	total_loss /= len(val_loader.dataset)
	loss_3d /= len(val_loader.dataset)
	loss_6d /= len(val_loader.dataset)
	loss_kld /= len(train_loader.dataset)

	return total_loss, loss_3d, loss_6d, loss_kld




def plot(epoch, X, pred, y,plot_samples=0):
	print(X.shape, pred.shape)
	if not os.path.isdir("./image"):
		os.mkdir("./image")

	if not os.path.isdir(os.path.join("./image",str(epoch))):
		os.mkdir(os.path.join("./image",str(epoch)))

	y = np.argmax(y, axis=1)

	N,T,P,J,_ = pred.shape
	pred = pred.reshape((N,T,J,3))
	X = X.reshape((N,T,J,3))
	for plot_ind,i in enumerate(np.random.choice(N,plot_samples)):
		fig = plt.figure(figsize=(8,4))
		ax = fig.add_subplot(111,projection='3d')
		ax.view_init(azim=-90,elev=-90)
		for j in range(T):
			plt.cla()
			gr_pose = X[i, j,:,:] - np.mean(X[i, j,:,:],axis=0,keepdims=True)

			pred_pose = pred[i, j,:,:] - np.mean(pred[i, j,:,:],axis=0,keepdims=True)
			pred_pose[:,0] += 0.7

			ax.scatter(gr_pose[:,0],gr_pose[:,1],gr_pose[:,2],s=5,c="green",label="Ground Truth")
			ax.scatter(pred_pose[:,0],pred_pose[:,1],pred_pose[:,2],s=5,c="red",label="Prediction")
			for jj,p in enumerate(parent_array):
				if p==-1:
					continue
				ax.plot([gr_pose[jj,0],gr_pose[p,0]],[gr_pose[jj,1],gr_pose[p,1]],[gr_pose[jj,2],gr_pose[p,2]],c="g",linewidth=3.0)
				ax.plot([pred_pose[jj,0],pred_pose[p,0]],[pred_pose[jj,1],pred_pose[p,1]],[pred_pose[jj,2],pred_pose[p,2]],c="r",linewidth=3.0)
			ax.legend()
			ax.axis('off')
			# plt.draw()
			plt.title(class_name[y[i]])
			plt.savefig("./image/" + str(epoch) + '/'  + str(plot_ind*10000+j) +".png")
		plt.close()	
	print("Plotting Complete")			


def infer(model, epoch, skeleton,num_samples=2,plot_samples=0):
	model.eval()
	skeleton = skeleton[:,0,:,:]
	plot_cls = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 30, 22, 33, 38]) - 1
	skeleton = torch.tensor(skeleton[np.random.choice(skeleton.shape[0],num_samples*plot_cls.shape[0])])
	y = np.array([ i for i in plot_cls for j in range(num_samples)])
	label = np.zeros((y.shape[0], num_class))
	label[np.arange(y.shape[0]), y] = 1
	label = torch.tensor(label).to(device).float()
	with torch.no_grad():
		m, v = model.gaussian_parameters(model.z_pre.squeeze(0), dim=0)
		idx = torch.distributions.categorical.Categorical(model.pi).sample((label.shape[0],))
		m, v = m[idx], v[idx]
		z = model.sample_gaussian(m, v)
		N = z.shape[0]
		z = torch.cat((z,label), dim=1)
		z = model.latent2hidden(z)
		z_body = z[:,:144] # for decoding body joints
		z_hand = z[:,144:] # for decoding hand joints
		z_body = z_body.reshape((N,4,-1))
		z_hand = z_hand.reshape((N,4,-1))
		
		x = model.decoder_net(z_body)
		hand_x = model.decoder_net_hand(z_hand)
		pred = torch.cat((x, hand_x), dim=2)
		pred_3d = fkt(pred[:,:,:].contiguous(), skeleton, device, parent_array)
		pred_3d = pred_3d.reshape((pred_3d.shape[0], pred_3d.shape[1], 52,-1)).cpu().data.numpy()
		print(pred_3d.shape)
		# sys.exit()
		
		if plot_samples > 0:
			plot_infer(epoch,y,pred_3d,num_samples,plot_samples)

	return pred_3d.reshape((y.shape[0],max_length,-1)),y


def plot_infer(epoch,y,X,num_samples,plot_samples):

	if not os.path.isdir(os.path.join("image_infer",str(epoch))):
		os.makedirs(os.path.join("image_infer",str(epoch)))
	# X = X[:,::2]
	N,T,J,_ = X.shape

	# specifying the width and the height of the box in inches
	fig = plt.figure(figsize=(9,6))
	ax = fig.add_subplot(111,projection='3d')

	ax.view_init(azim=90,elev=90)
	for i in range(0,N):
		name = class_name[y[i]]
		gr_pose = X[i:i+plot_samples] 
		gr_pose[...,0] +=  3*np.arange(-(plot_samples//2),(plot_samples+1)//2).reshape((plot_samples,1,1))
		min_lim = np.min(gr_pose,axis=(0,1,2)).min()
		max_lim = np.max(gr_pose,axis=(0,1,2)).max()
		for j in range(T):
			plt.cla()
			for k in range(plot_samples):
				ax.scatter(gr_pose[k,j,:,0],gr_pose[k,j,:,1],gr_pose[k,j,:,2],s=10,c="green")
			
			for jj,p in enumerate(parent_array):
				if p == -1:
					continue
				for k in range(plot_samples):	
					ax.plot([gr_pose[k,j,jj,0],gr_pose[k,j,p,0]],[gr_pose[k,j,jj,1],gr_pose[k,j,p,1]],[gr_pose[k,j,jj,2],gr_pose[k,j,p,2]],c="green",linewidth=3.0)


			ax.set_xlim(min_lim, max_lim)
			ax.set_ylim(min_lim, max_lim)
			ax.set_zlim(min_lim, max_lim)

			# ax.legend()
			ax.axis('off')
			# plt.draw()

			plt.title("Example: {}(timestep: {})".format(name, j+1))
			plt.savefig(os.path.join("image_infer/" ,str(epoch), str(i*1000000+j) +".png"), pad_inches=0)
	plt.close()	
	print("Plotting Complete")

def get_datasets(main_path, batch_size, num_workers):
	# datadir = os.path.join(main_path,'data')
	path = os.path.join(os.path.join(main_path, "data", 'NTU-X-120-train.h5'))
	f = h5py.File(path, 'r')

	# train_dataset = NTUDataset((f['TransformedJoints'][:],f['6dRs'][:],f['Joints'][:],f['Mask'][:],f['y'][:]))
	print("loading dataset")
	x = f['x'][:]
	print("x loaded..")
	pose = f['pose'][:]
	print("pose loaded..")
	mean_pose = f['mean_pose'][:]
	print("mean_pose loaded..")
	y = f['y'][:]
	print("y loaded..")
	root = f['root'][:]
	train_dataset = NTUDataset((x,pose,mean_pose,y, root))
	print(f"Train Dataset Loaded {train_dataset.N} samples")

	# test_dataset = NTUDataset((f['test-TransformedJoints'][:],f['test-6dRs'][:],f['test-Joints'][:],f['test-Mask'][:],f['test-y'][:]))
	# test_dataset = NTUDataset((f['test_x'][:],f['test_pose'][:],f['test_mean_pose'][:],f['test_mask'][:],f['test_y'][:]))
	# print(f"Test Dataset Loaded {test_dataset.N} samples")

	print("Done..")
	train_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=num_workers,shuffle=True)
	# val_loader = DataLoader(test_dataset,batch_size=batch_size, num_workers=num_workers,shuffle=True)
	return train_loader, train_dataset.N, mean_pose



def frange_cycle_linear(n_iter, start=0.0, stop=0.1,  n_cycle=5, ratio=0.6):
	L = np.ones(n_iter) * stop
	period = n_iter/n_cycle
	step = (stop-start)/(period*ratio) # linear schedule

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i+c*period) < n_iter):
			L[int(i+c*period)] = v
			v += step
			i += 1
	return L


def save_model(model,epoch):
	if not os.path.isdir(os.path.join(main_path, "./checkpoints_MUGL++_120_seq_self_attn_end")):
		os.mkdir(os.path.join(main_path, "./checkpoints_MUGL++_120_seq_self_attn_end"))
	
	filename = os.path.join(main_path, 'checkpoints_MUGL++_120_seq_self_attn_end', 'model_{}.pt'.format(epoch))
	torch.save(model.state_dict(), filename)


if __name__ == '__main__':
	train_loader, N, skeleton = get_datasets(main_path, batch_size, num_workers)
	model = Model(num_class, latent_dim).to(device)
	total_params = sum(p.numel() for p in model.parameters())
	print('Total number of parameters:', total_params)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)
	scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
	L1Loss = nn.L1Loss().to(device)

	train_loss_list = []
	test_loss_list = []

	# Cyclic Annealing
	no_itr = math.ceil(N/batch_size)
	L = frange_cycle_linear(no_itr)
	print(L)
	if load_epoch > 0:
		model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))


	for epoch in range(0, max_epochs):
		if epoch > load_epoch:
			model.train()
			train_loss, train_recon, train_6d, train_kld, train_root, train_seq = train(epoch, model, train_loader, optimizer, L1Loss, L)
			# with torch.no_grad():
			# 	model.eval()
			# 	test_loss, test_recon, test_6d, test_kld = test(epoch, model, val_loader, L1Loss,plot_samples=1)
				# if epoch  % 30 == 0:
				# 	num_samples = 50
				# 	pred_3d,pred_y = infer(model, epoch, skeleton,num_samples=num_samples,plot_samples=10)
				# 	real_3d = val_loader.dataset.get_samples(pred_y)
				# 	print("MMD Score:",mmd_function(pred_3d,real_3d,pred_y))					
			for param_group in optimizer.param_groups:
				print('Learning Rate:',param_group['lr'])
			print('Train Epoch:{}/{} Train_loss: {} Recon: {} Rot: {} kld: {} root: {} seq: {}'.format(epoch, max_epochs, train_loss, train_recon, train_6d, train_kld, train_root, train_seq))
			# print('Test Epoch:{}/{} Train_loss: {} Recon: {} Rot: {} kld: {}'.format(epoch, max_epochs, test_loss, test_recon, test_6d, test_kld))
		

		if epoch < 170:
			scheduler.step()

		if epoch  > 100:
			save_model(model, epoch)






