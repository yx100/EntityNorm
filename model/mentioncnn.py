import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt

class CNN(nn.Module):
	def __init__(self, vocab, num_classes):

		super(CNN, self).__init__()

		self.embedding = vocab.init_embed_layer()
		self.hidden_size = opt.hidden_size
		D = self.embedding.weight.size(1)
		self.hidden_size = opt.hidden_size
		Ci =1
		Co = opt.kernel_num
		Ks = [int(k) for k in list(opt.kernel_sizes) if k != ","]

		# self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
		# self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

		self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
									 padding=(K // 2, 0), dilation=1, bias=False) for K in Ks])
		'''
		self.conv13 = nn.Conv2d(Ci, Co, (2, D))
		self.conv14 = nn.Conv2d(Ci, Co, (3, D))
		self.conv15 = nn.Conv2d(Ci, Co, (4, D))
		'''
		self.dropout = nn.Dropout(opt.dropout)
		self.hidden = nn.Linear(len(Ks) * Co, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, num_classes)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, x):
		inputs, lengths, seq_recover = x
		x = self.embedding(inputs)# (N, W, D)
		x = x.unsqueeze(1)  # (N, Ci, W, D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
		x = torch.cat(x, 1)

		'''
		x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
		x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
		x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
		x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
		'''
		x = self.dropout(x)  # (N, len(Ks)*Co)
		hidden = self.hidden(x)  # (N, hidden)
		output = self.out(hidden)
		return output
