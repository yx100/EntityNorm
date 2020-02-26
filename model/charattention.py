import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class CharATTEN(nn.Module):

	def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu):
		super(CharATTEN, self).__init__()
		print("build char sequence feature extractor: ATTENTION ...")
		self.gpu = gpu
		self.hidden_dim = hidden_dim
		self.embed_size = embedding_dim

		self.char_drop = nn.Dropout(dropout)
		self.char_embeddings = nn.Embedding(alphabet_size, self.embed_size)
		if pretrain_char_embedding is not None:
			self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
		else:
			self.char_embeddings.weight.data.copy_(
				torch.from_numpy(self.random_embedding(alphabet_size, self.embed_size)))
		self.char_W = nn.Linear(self.embed_size, 1, bias=False)
		self.hidden = nn.Linear(self.embed_size, self.hidden_dim)
		if torch.cuda.is_available():
			self.char_drop = self.char_drop.cuda(self.gpu)
			self.char_embeddings = self.char_embeddings.cuda(self.gpu)
			self.char_W = self.char_W.cuda(self.gpu)
	def random_embedding(self, vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb

	def forward(self, input, seq_lengths):
		char_embeds = self.char_drop(self.char_embeddings(input))
		batch_size, max_len, _ = char_embeds.size()

		flat_input = char_embeds.contiguous().view(-1, self.embed_size)
		logits = self.char_W(flat_input).view(batch_size, max_len)
		alphas = functional.softmax(logits, dim=1)

		# computing mask
		idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda(self.gpu)
		mask = autograd.Variable((idxes < seq_lengths.unsqueeze(1)).float())

		alphas = alphas * mask
		# renormalize
		alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
		atten_out = torch.bmm(alphas.unsqueeze(1), char_embeds).squeeze(1)
		hidden = self.hidden(atten_out)

		return hidden
