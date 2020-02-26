import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
from options import opt
from model.charcnn import CharCNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttenCNN_SentLSTM(nn.Module):
	def __init__(self, vocab, num_classes, char_alphabet):
		super(AttenCNN_SentLSTM,self).__init__()
		self.embed_size = opt.word_emb_size
		self.embedding = vocab.init_embed_layer()
		self.hidden_size = opt.hidden_size
		self.char_hidden_dim = 10
		self.char_embedding_dim = 20
		self.char_feature = CharCNN(len(char_alphabet), None, self.char_embedding_dim, self.char_hidden_dim,
									opt.dropout, opt.gpu)
		self.input_size = self.embed_size + self.char_hidden_dim

		self.W = nn.Linear(self.input_size, 1, bias=False)

		self.mention_hidden = nn.Linear(self.input_size, self.hidden_size)

		#sentence lstm
		self.lstm_hidden = opt.hidden_size
		self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden, num_layers=1, batch_first=True,
							bidirectional=True)
		self.sent_hidden_size = opt.sent_hidden_size
		self.sent_hidden = nn.Linear(self.lstm_hidden*2, self.sent_hidden_size)
		self.hidden = nn.Linear(self.hidden_size + self.sent_hidden_size, self.hidden_size)  # mention_hidden_size + sentence_hidden_size
		self.out = nn.Linear(self.hidden_size, num_classes)
		self.dropout = nn.Dropout(opt.dropout)

	def conv_and_pool(self, x, conv):
		x = functional.relu(conv(x)).squeeze(3)  # (N, Co, W)
		x = functional.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, input, char_inputs,sent_inputs):
		"""
		inputs: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
		"""

		entity_words, entity_lengths, entity_seq_recover = input
		entity_words = autograd.Variable(entity_words)
		entity_words_embeds = self.embedding(entity_words)
		batch_size, max_len, _ = entity_words_embeds.size()

		char_inputs, char_seq_lengths, char_seq_recover = char_inputs
		char_features = self.char_feature.get_last_hiddens(char_inputs)
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

		mention_hidden = self.mention_hidden(atten_input)

		sent_inputs, sent_seq_lengths = sent_inputs
		sent_embedding = self.embedding(sent_inputs)
		packed_words = pack_padded_sequence(sent_embedding, sent_seq_lengths.cpu().numpy(), True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out)
		hid_size = lstm_out.size(2) // 2
		sents_bilstm_out = torch.cat([lstm_out[0, :, :hid_size], lstm_out[-1, :, hid_size:]],
									 dim=1)
		sent_hidden = self.sent_hidden(sents_bilstm_out)

		x = torch.cat((mention_hidden, sent_hidden), 1)
		x = self.dropout(x)  # (N, len(Ks)*Co)
		hidden = self.hidden(x)  # (N, hidden)
		output = self.out(hidden)
		return output


