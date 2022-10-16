import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sys




class BasicBlock(nn.Module):
	"""
	Basic block is composed of 2 CNN layers with residual connection.
	Each CNN layer is followed by batchnorm layer and swish activation 
	function. 
	Args:
		in_channel: number of input channels
		out_channel: number of output channels
		k: (default = 1) kernel size
	"""
	def __init__(self, in_channel, out_channel, k=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_channel,
			out_channel,
			kernel_size=k,
			padding=(0, 0),
			stride=(1, 1))
		self.bn1 = nn.BatchNorm2d(out_channel)

		self.conv2 = nn.Conv2d(
			out_channel,
			out_channel,
			kernel_size=1,
			padding=(0, 0),
			stride=(1, 1))
		self.bn2 = nn.BatchNorm2d(out_channel)

		self.shortcut = nn.Sequential()
		# if in_channel != out_channel:
		self.shortcut.add_module(
			'conv',
			nn.Conv2d(
				in_channel,
				out_channel,
				kernel_size=k,
				padding=(0,0),
				stride=(1,1)))
		self.shortcut.add_module('bn', nn.BatchNorm2d(out_channel))

	def swish(self,x):
		"""
		We use swish in spatio-temporal encoding/decoding. We tried with 
		other activation functions such as ReLU and LeakyReLU. But we 
		achieved the best performance with swish activation function.
		Args:
			X: tensor: (batch_size, ...)
		Return:
			_: tensor: (batch, ...): applies swish 
			activation to input tensor and returns  
		"""
		return x*torch.sigmoid(x)

	def forward(self, x):
		y = self.swish(self.conv1(x))
		y = self.swish(self.conv2(y))
		y = y + self.shortcut(x)
		y = self.swish(y)
		return y


class BasicBlockTranspose(nn.Module):
	"""
	Basic block is composed of 2 CNN layers with residual connection.
	Each CNN layer is followed by batchnorm layer and swish activation 
	function. 
	Args:
		in_channel: number of input channels
		out_channel: number of output channels
		k: (default = 1) kernel size
	"""
	def __init__(self, in_channel, out_channel, k=(1,1)):
		super(BasicBlockTranspose, self).__init__()
		self.conv1 = nn.ConvTranspose2d(
			in_channel,
			out_channel,
			kernel_size=k,
			padding=(0, 0),
			stride=(1, 1))
		self.bn1 = nn.BatchNorm2d(out_channel)

		self.conv2 = nn.ConvTranspose2d(
			out_channel,
			out_channel,
			kernel_size=1,
			padding=(0, 0),
			stride=(1, 1))
		self.bn2 = nn.BatchNorm2d(out_channel)

		self.shortcut = nn.Sequential()
		# if in_channel != out_channel:
		self.shortcut.add_module(
			'conv',
			nn.ConvTranspose2d(
				in_channel,
				out_channel,
				kernel_size=k,
				padding=(0,0),
				stride=(1,1)))
		self.shortcut.add_module('bn', nn.BatchNorm2d(out_channel))

	def swish(self,x):
		"""
		We use swish in spatio-temporal encoding/decoding. We tried with 
		other activation functions such as ReLU and LeakyReLU. But we 
		achieved the best performance with swish activation function.
		Args:
			X: tensor: (batch_size, ...)
		Return:
			_: tensor: (batch, ...): applies swish 
			activation to input tensor and returns  
		"""
		return x*torch.sigmoid(x)

	def forward(self, x):
		y = self.swish(self.bn1(self.conv1(x)))
		y = self.swish(self.bn2(self.conv2(y)))
		y = y + self.shortcut(x)
		y = self.swish(y)
		return y



class Self_Attn_Seq(nn.Module):
    def __init__(self,in_dim, n_head=3):
        super(Self_Attn_Seq,self).__init__()
        input_dim = in_dim
        self.n_head = n_head # number of attenn head
        self.hidden_size_attention = input_dim // self.n_head
        self.w_q = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        self.w_k = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        self.w_v = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        nn.init.normal_(self.w_k.weight, mean=0,
                        std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        nn.init.normal_(self.w_v.weight, mean=0,
                        std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        self.temperature = np.power(self.hidden_size_attention, 0.5)

        self.softmax = nn.Softmax(dim=2)
        self.linear2 = nn.Linear(self.n_head * self.hidden_size_attention, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
    

    def forward(self, q):
        n_head = self.n_head
        residual = q
        k, v = q, q
        bs, len, _ = q.size()
        q = self.w_q(q).view(bs, len, n_head, self.hidden_size_attention)
        k = self.w_k(k).view(bs, len, n_head, self.hidden_size_attention)
        v = self.w_v(v).view(bs, len, n_head, self.hidden_size_attention)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)

        # generate mask
        subsequent_mask = torch.triu(
            torch.ones((len, len), device=q.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(bs, -1, -1).gt(0)
        mask = subsequent_mask.repeat(n_head, 1, 1)

        # self attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        output = output.view(n_head, bs, len, self.hidden_size_attention)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, len, -1)
        output = self.gamma * self.linear2(output) + residual


        attn = attn.view(n_head,bs,len,len)
        attn_avg = torch.mean(attn,0)
        return output, attn_avg





class Model(nn.Module):
	def __init__(self, num_class, latent_dim, components=120):
		super(Model, self).__init__()
		self.latent_dim = latent_dim

		# encoder
		self.encoder1 = BasicBlock(1,1,k=3)
		self.encoder2 = BasicBlock(1,1,k=3)

		self.encoder1_hand = BasicBlock(1,1,k=3)
		self.encoder2_hand = BasicBlock(1,1,k=3)
		self.encoder_attn0 = Self_Attn_Seq(144)

		# self attention layer
		self.encoder_attn1 = Self_Attn_Seq(168) # hidden size 40
		self.encoder_attn2 = Self_Attn_Seq(80) # hidden size 40
		self.encoder_attn1_hand = Self_Attn_Seq(232) # hidden size 40
		self.encoder_attn2_hand = Self_Attn_Seq(112) # hidden size 40
		self.encode_t = BasicBlock(62, 32)
		self.encode_t0 = BasicBlock(32, 16)
		self.encode_t1 = BasicBlock(14, 8)
		self.encode_t2 = BasicBlock(8, 4)

		self.encode_s1 = BasicBlock(42, 42, k=(3,1))
		self.encode_s2 = BasicBlock(40, 40, k=(3,1))
		# self.encode_s3 = BasicBlock(22, 22, k=(3,1))

		self.encode_t1_hand = BasicBlock(62, 32)
		self.encode_t2_hand = BasicBlock(32, 16)
		self.encode_t1_hand1 = BasicBlock(14, 8)
		self.encode_t2_hand1 = BasicBlock(8, 4)
		# self.encode_t3_hand = BasicBlock(8, 4)

		self.encode_hand_s1 = BasicBlock(58, 58, k=(3,1))
		self.encode_hand_s2 = BasicBlock(56, 56, k=(3,1))

		# decoder 
		self.conv1 = BasicBlock(1,1)
		self.conv2 = BasicBlock(1,1)
		self.conv3 = BasicBlock(1,1)
		self.conv4 = BasicBlock(1,1)
		self.decode_t = BasicBlock(4,8)
		self.decode_t1 = BasicBlock(8,14)
		self.decode_t2 = BasicBlock(16,32)
		self.decode_t3 = BasicBlock(32,62)
		
		self.decode_t_hand = BasicBlock(4,8)
		self.decode_t1_hand = BasicBlock(8,14)
		self.decode_t_hand1 = BasicBlock(16,32)
		self.decode_t1_hand1 = BasicBlock(32,62)
		# self.decode_t2_hand = BasicBlock(32,64)

		# self attention layer
		self.decoder_attn1 = Self_Attn_Seq(80)
		self.decoder_attn2 = Self_Attn_Seq(168)
		self.decoder_attn1_hand = Self_Attn_Seq(112)
		self.decoder_attn2_hand = Self_Attn_Seq(58*4)
		self.decoder = nn.Linear(80, 168)
		self.decoder1 = nn.Linear(168,264)
		self.decoder_hand = nn.Linear(112,58*4)
		self.decoder_hand1 = nn.Linear(58*4,60*6)

		self.decode_s1 = BasicBlockTranspose(40, 40, k=(3,1))
		self.decode_s2 = BasicBlockTranspose(42, 42, k=(3,1))
		self.decode_s1_hand = BasicBlockTranspose(56, 56, k=(3,1))
		self.decode_s2_hand = BasicBlockTranspose(58, 58, k=(3,1))
		# self.decode_s3 = BasicBlockTranspose(22, 22, k=(3,1))

		# root trajectory
		self.root1 = nn.Conv1d(4,8,1)
		self.root2 = nn.Conv1d(8,16,1)
		self.root3 = nn.Conv1d(16,32,1)
		self.root4 = nn.Conv1d(32,64,1)
		self.root5 = nn.Linear(80,3*2)

		# root trajectory encoder
		self.r_encoder0 = nn.Conv1d(64, 32, 1)
		self.r_encoder1 = nn.Conv1d(32, 16, 1)
		self.r_encoder2 = nn.Conv1d(16, 8, 1)
		self.r_encoder3 = nn.Conv1d(8, 4, 1)
		self.r_encoder4 = nn.Linear(24, 20)
		# self.r_encoder4 = nn.Linear(32, 16, 5)

		# sequence length encoder
		self.seq_encoder0 = nn.Conv1d(64,32,1)
		self.seq_encoder1 = nn.Conv1d(32,16,1)
		self.seq_encoder2 = nn.Conv1d(16,8,1)
		self.seq_encoder3 = nn.Conv1d(8,4,1)
		self.seq_encoder4 = nn.Linear(4, 4)

		# sequence length decoder
		# decoder
		self.seq_decoder1 = nn.Linear(latent_dim, 4)
		self.seq_decoder2 = nn.Conv1d(4,8,1)
		self.seq_decoder3 = nn.Conv1d(8,16,1)
		self.seq_decoder4 = nn.Conv1d(16,32,1)
		self.seq_decoder5 = nn.Conv1d(32,64,1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		# wrist position encoder
		self.wrist_encoder0 = nn.Conv1d(64, 32, 1)
		self.wrist_encoder1 = nn.Conv1d(32, 16, 1)
		self.wrist_encoder2 = nn.Conv1d(16, 8, 1)
		self.wrist_encoder3 = nn.Conv1d(8, 4, 1)
		self.wrist_encoder4 = nn.Linear(48, 10)

		self.hidden2latent = nn.Linear(self.latent_dim+20+10+num_class+6+4, self.latent_dim*2)
		self.latent2hidden = nn.Linear(self.latent_dim+num_class+6, self.latent_dim)

		# Gausiam mixture parameters
		self.components = components
		# mixture of Gaussian parameters
		self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.components, self.latent_dim)
										/ np.sqrt(self.components * self.latent_dim))
		# Uniform weighting
		self.pi = torch.nn.Parameter(torch.ones(components) / components, requires_grad=False)


	def swish(self,x):
		"""
		We use swish in spatio-temporal encoding/decoding. We tried with 
		other activation functions such as ReLU and LeakyReLU. But we 
		achieved the best performance with swish activation function.
		Args:
			X: tensor: (batch_size, ...)
		Return:
			_: tensor: (batch, ...): applies swish 
			activation to input tensor and returns  
		"""
		return x*torch.sigmoid(x)


	def encoder_net(self, X):
		"""
		Encoder first downsamples the input motion in the spatial dimension
		and then downsamples in the temporal dimension and returns spatio-
		temporal feature.
		Args:
			X: tensor: (batch_size, 32, 48, 6): input motion of 2 persons. 24
			joints for each persons so total 48 joints.
		Return:
			x: tensor: (batch_size, 4, ...): spatio-temporal feature 
		"""
		N,T,J = X.shape
		# pose encoding
		x = X.reshape((N*T,1,44,6))
		x = self.encoder1(x)

		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.reshape((N,T,42,4))
		x = x.transpose(2,1)
		x = self.encode_s1(x)
		x = x.transpose(2,1) # (b, 56, 20, 2)
		# ----------------------------------------------------------------

		# temporal encoding
		x = self.encode_t(x)
		x = self.encode_t0(x)
		x = x.reshape((N,16,-1))
		# ------------------------ End of block one ---------------------


		N,T,J = x.shape
		x, attn = self.encoder_attn1(x)
		x = x.reshape((N*T,1,42,4))
		x = self.encoder2(x)

		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.reshape((N,T,40,2))
		x = x.transpose(2,1)
		x = self.encode_s2(x)
		x = x.transpose(2,1) # (b, 56, 20, 2)

		# ----------------------------------------------------------------
		x = self.encode_t1(x)
		x = self.encode_t2(x)
		x = x.reshape((N,4,-1))
		x, attn = self.encoder_attn2(x)
		# ------------------------ End of block two ---------------------

		return x

	def decoder_net(self, X):
		"""
		The deocder is opposit of the encoder. It takes the vector sampled
		from a mixture of gaussian parameter conditioned by class label on-
		hot vector and viewpoint vector, upsamples it in the temporal dimension 
		first and then upsamples it in the spatial dimension.
		Args:
			X: tensor: (batch_size, 4, ...): sampled vector conditionied on class 
			label and viewpoint
		Return:
			x: tensor: (batch_size, 32, 48, 6): generated human motion
		"""
		N,T,J = X.shape
		x, attn = self.decoder_attn1(X)
		# temporal decoding
		x = x.reshape((N,T,40,2))
		x = self.decode_t(x)
		x = self.decode_t1(x)

		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.transpose(2,1)
		x = self.decode_s1(x)
		x = x.transpose(2,1)
		# ----------------------------------------------------------------
		# pose decoding
		x = x.reshape((N*16,1,40,2))
		x = self.conv1(x)
		x = x.reshape((N,16, -1))

		x = self.decoder(x)
		x, attn = self.decoder_attn2(x)
		# ------------------------ End of block one ---------------------

		N,T,J = x.shape
		# temporal decoding
		x = x.reshape((N,T,42, 4))
		x = self.decode_t2(x)
		x = self.decode_t3(x)

		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.transpose(2,1)
		x = self.decode_s2(x)
		x = x.transpose(2,1)
		# ----------------------------------------------------------------
		# pose decoding
		x = x.reshape((N*64,1,42,4))
		x = self.conv2(x)
		x = x.reshape((N,64, -1))
		x = self.decoder1(x)
		# ------------------------ End of block two ---------------------

		return x


	def encoder_net_hand(self, X):
		"""
		Encoder first downsamples the input motion in the spatial dimension
		and then downsamples in the temporal dimension and returns spatio-
		temporal feature.
		Args:
			X: tensor: (batch_size, 32, 48, 6): input motion of 2 persons. 24
			joints for each persons so total 48 joints.
		Return:
			x: tensor: (batch_size, 4, ...): spatio-temporal feature 
		"""
		N,T,J = X.shape
		# pose encoding
		# x, attn = self.encoder_attn0_hand(X)
		x = X.reshape((N*T,1,60,6))
		x = self.encoder1_hand(x)

		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.reshape((N,T,58,4))
		x = x.transpose(2,1)
		x = self.encode_hand_s1(x)
		x = x.transpose(2,1) # (b, 56, 20, 2)
		# ----------------------------------------------------------------

		# temporal encoding
		x = x.reshape((N,62,58,4))
		# x = self.encode_t(x)
		x = self.encode_t1_hand(x)
		x = self.encode_t2_hand(x)
		# x = self.encode_t3_hand(x)
		x = x.reshape((N, 16, -1))
		x, attn = self.encoder_attn1_hand(x)
		# ------- Encoding block 1 ---------------------------------------


		x = x.reshape((N*16,1,58,4))
		x = self.encoder2_hand(x)
		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.reshape((N,16,56,2))
		x = x.transpose(2,1)
		x = self.encode_hand_s2(x)
		x = x.transpose(2,1) # (b, 56, 20, 2)
		# ----------------------------------------------------------------

		# temporal encoding
		x = x.reshape((N,14,56,2))
		# x = self.encode_t(x)
		x = self.encode_t1_hand1(x)
		x = self.encode_t2_hand1(x)
		x = x.reshape((N,4,-1))
		x, attn = self.encoder_attn2_hand(x)
		return x



	def decoder_net_hand(self, X):
		"""
		The deocder is opposit of the encoder. It takes the vector sampled
		from a mixture of gaussian parameter conditioned by class label on-
		hot vector and viewpoint vector, upsamples it in the temporal dimension 
		first and then upsamples it in the spatial dimension.
		Args:
			X: tensor: (batch_size, 4, ...): sampled vector conditionied on class 
			label and viewpoint
		Return:
			x: tensor: (batch_size, 32, 48, 6): generated human motion
		"""
		N,T,J = X.shape
		# temporal decoding
		x, attn = self.decoder_attn1_hand(X)
		x = x.reshape((N,T,56,2))
		x = self.decode_t_hand(x)
		x = self.decode_t1_hand(x)
		
		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.transpose(2,1)
		x = self.decode_s1_hand(x)
		x = x.transpose(2,1)
		# ----------------------------------------------------------------

		# pose decoding
		x = x.reshape((N*16,1,56,2))
		x = self.conv3(x)
		x = x.reshape((N,16, -1))
		x = self.decoder_hand(x)
		# --------------- End of block 1 ---------------------------------


		x, attn = self.decoder_attn2_hand(x)
		x = x.reshape((N,16,58,4))
		x = self.decode_t_hand1(x)
		x = self.decode_t1_hand1(x)
	
		# ----------------------------------------------------------------
		# ------------------------- newly added --------------------------
		# ----------------------------------------------------------------
		x = x.transpose(2,1)
		x = self.decode_s2_hand(x)
		x = x.transpose(2,1)
		# ----------------------------------------------------------------
		# pose decoding
		x = x.reshape((N*64,1,58,4))
		x = self.conv4(x)
		x = x.reshape((N,64, -1))
		x = self.decoder_hand1(x)

		return x


	def root_traj(self, z):
		"""
		This function calculate the root trajectory for 2-person inteaction
		classes. generates the displacement of the second person's root from the
		first person's root.
		Args:
			z: tensor: (batch_size, 4, ...): sampled vector conditionied on class 
			label and viewpoint
		Return:
			z: tensor: (batch_size, 32, 3): displacement
		"""
		N,_ = z.shape
		z = z.reshape((N, 4, -1))
		z = self.swish(self.root1(z))
		z = self.swish(self.root2(z))
		z = self.swish(self.root3(z))
		z = self.swish(self.root4(z))
		z = self.root5(z)
		return z


	def root_traj_encoder(self, root):
		root = root.float()
		z = self.swish(self.r_encoder0(root))
		z = self.swish(self.r_encoder1(z))
		z = self.swish(self.r_encoder2(z))
		z = self.swish(self.r_encoder3(z))
		z = z.reshape((z.shape[0], -1))
		z = self.r_encoder4(z)
		return z


	def seq_encoder(self, x):
		# x = x[:,:,None]
		N,T,_ = x.shape
		z = self.relu(self.seq_encoder0(x))
		z = self.relu(self.seq_encoder1(z))
		z = self.relu(self.seq_encoder2(z))
		z = self.relu(self.seq_encoder3(z))
		z = z.reshape((N, -1))
		z = self.relu(self.seq_encoder4(z))
		return z


	def seq_decoder(self, z):
		N,_ = z.shape
		z = self.relu(self.seq_decoder1(z))
		z = z.unsqueeze(-1)
		z = self.relu(self.seq_decoder2(z))
		z = self.relu(self.seq_decoder3(z))
		z = self.relu(self.seq_decoder4(z))
		z = self.sigmoid(self.seq_decoder5(z))
		return z


	def wrist_position_encoder(self, root):
		root = root.float()
		z = self.swish(self.wrist_encoder0(root))
		z = self.swish(self.wrist_encoder1(z))
		z = self.swish(self.wrist_encoder2(z))
		z = self.swish(self.wrist_encoder3(z))
		z = z.reshape((z.shape[0], -1))
		z = self.wrist_encoder4(z)
		return z


	def forward(self, x, hand, y, rot, root, seq, wrist):
		N,T,J = x.shape
		z = self.encoder_net(x)
		z_hand = self.encoder_net_hand(hand)
		z = z.reshape((N,-1))
		z_hand = z_hand.reshape((N,-1))

		root_encoding = self.root_traj_encoder(root)
		seq = self.seq_encoder(seq)
		wrist_encoding = self.wrist_position_encoder(wrist)
		z = torch.cat((z, z_hand, root_encoding, wrist_encoding, seq, rot, y.float()), dim=1)

		z = self.hidden2latent(z)
		mean, var = self.gaussian_parameters(z, dim=1)

		# Gaussian mixture
		prior = self.gaussian_parameters(self.z_pre, dim=1)

		z = self.sample_gaussian(mean, var)

		# terms for KL divergence
		log_q_phi = self.log_normal(z, mean, var)
		log_p_theta = self.log_normal_mixture(z, prior[0], prior[1])
		kld = torch.mean(log_q_phi - log_p_theta)

		# z = self.reparameterization(mean, logvar)
		z = torch.cat((z,rot, y.float()), dim=1)
		z = self.latent2hidden(z)

		# decoding phase
		seq_pred = self.seq_decoder(z)
		
		z_body = z[:,:320] # for decoding body joints
		z_hand = z[:,320:] # for decoding hand joints
		root = self.root_traj(z_body)
		z_body = z_body.reshape((N,4,-1))
		z_hand = z_hand.reshape((N,4,-1))
		
		x = self.decoder_net(z_body)
		hand_x = self.decoder_net_hand(z_hand)
		x = x.reshape(N, T, 2, 22, -1)
		hand_x = hand_x.reshape((N, T, 2, 30, -1))
		
		return x, hand_x, kld, root, seq_pred

	def sample_gaussian(self, m, v):
		"""
		Element-wise application reparameterization trick to sample from Gaussian
		Args:
			m: tensor: (batch, ...): Mean
			v: tensor: (batch, ...): Variance
		Return:
			z: tensor: (batch, ...): Samples
		"""
		sample = torch.randn(m.shape).to(m.device)
		

		z = m + (v**0.5)*sample
		return z



	def gaussian_parameters(self, h, dim=-1):
		"""
		Converts generic real-valued representations into mean and variance
		parameters of a Gaussian distribution
		Args:
			h: tensor: (batch, ..., dim, ...): Arbitrary tensor
			dim: int: (): Dimension along which to split the tensor for mean and
				variance
		Returns:z
			m: tensor: (batch, ..., dim / 2, ...): Mean
			v: tensor: (batch, ..., dim / 2, ...): Variance
		"""
		m, h = torch.split(h, h.size(dim) // 2, dim=dim)
		v = F.softplus(h) + 1e-8
		return m, v



	def log_normal(self, x, m, v):
		"""
		Computes the elem-wise log probability of a Gaussian and then sum over the
		last dim. Basically we're assuming all dims are batch dims except for the
		last dim.
		Args:
			x: tensor: (batch, ..., dim): Observation
			m: tensor: (batch, ..., dim): Mean
			v: tensor: (batch, ..., dim): Variance
		Return:
			kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
				that the summation dimension (dim=-1) is not kept
		"""

		const = -0.5*x.size(-1)*torch.log(2*torch.tensor(np.pi))
		log_det = -0.5*torch.sum(torch.log(v), dim = -1)
		log_exp = -0.5*torch.sum( (x - m)**2/v, dim = -1)
		log_prob = const + log_det + log_exp

		return log_prob


	def log_normal_mixture(self, z, m, v):
		"""
		Computes log probability of a uniformly-weighted Gaussian mixture.
		Args:
			z: tensor: (batch, dim): Observations
			m: tensor: (batch, mix, dim): Mixture means
			v: tensor: (batch, mix, dim): Mixture variances
		Return:
			log_prob: tensor: (batch,): log probability of each sample
		"""
		z = z.unsqueeze(1)
		log_probs = self.log_normal(z, m, v)
		log_prob = self.log_mean_exp(log_probs, 1)

		return log_prob

	def log_mean_exp(self, x, dim):
		"""
		Compute the log(mean(exp(x), dim)) in a numerically stable manner
		Args:
			x: tensor: (...): Arbitrary tensor
			dim: int: (): Dimension along which mean is computed
		Return:
			_: tensor: (...): log(mean(exp(x), dim))
		"""
		return self.log_sum_exp(x, dim) - np.log(x.size(dim))


	def log_sum_exp(self, x, dim=0):
		"""
		Compute the log(sum(exp(x), dim)) in a numerically stable manner
		Args:
			x: tensor: (...): Arbitrary tensor
			dim: int: (): Dimension along which sum is computed
		Return:
			_: tensor: (...): log(sum(exp(x), dim))
		"""
		max_x = torch.max(x, dim)[0]
		new_x = x - max_x.unsqueeze(dim).expand_as(x)
		return max_x + (new_x.exp().sum(dim)).log()

