import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
from options import opt
from model.charattention import CharATTEN

class AttenATTEN(nn.Module):
	def __init__(self, vocab, num_classes, char_alphabet):
		super(AttenATTEN,self).__init__()
		self.embed_size = opt.word_emb_size
		self.embedding = vocab.init_embed_layer()
		self.hidden_size = opt.hidden_size
		self.char_hidden_dim = 10
		self.char_embedding_dim = 20
		self.char_feature = CharATTEN(len(char_alphabet), None, self.char_embedding_dim, self.char_hidden_dim,
									opt.dropout, opt.gpu)
		self.input_size = self.embed_size + self.char_hidden_dim

		self.W = nn.Linear(self.input_size, 1, bias=False)

		self.hidden = nn.Linear(self.input_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, num_classes)
		self.dropout = nn.Dropout(opt.dropout)

	def forward(self, input, char_inputs):
		"""
		inputs: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
		"""

		entity_words, entity_lengths, entity_seq_recover = input
		entity_words = autograd.Variable(entity_words)
		entity_words_embeds = self.embedding(entity_words)
		batch_size, max_len, _ = entity_words_embeds.size()

		char_inputs, char_seq_lengths, char_seq_recover = char_inputs
		char_features = self.char_feature(char_inputs, char_seq_lengths)
		char_features = char_features[char_seq_recover]
		char_features = char_features.view(batch_size, max_len, -1)

		input_embeds = torch.cat((entity_words_embeds, char_features), 2)

		flat_input = input_embeds.contiguous().view(-1, self.input_size)
		logits = self.W(flat_input).view(batch_size, max_len)
		alphas = functional.softmax(logits, dim=1)

		# computing mask
		idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda(opt.gpu)
		mask = autograd.Variable((idxes < entity_lengths.unsqueeze(1)).float())

		alphas = alphas * mask
		alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
		atten_input = torch.bmm(alphas.unsqueeze(1), input_embeds).squeeze(1)
		atten_input = self.dropout(atten_input)

		hidden = self.hidden(atten_input)
		output = self.out(hidden)
		return output


