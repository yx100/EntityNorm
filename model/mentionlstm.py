import torch
import torch.nn as nn
from options import opt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        super(LSTM,self).__init__()
        self.embed_size = opt.word_emb_size
        self.embedding = vocab.init_embed_layer()
        self.hidden_size = opt.hidden_size

        self.lstm_hidden = opt.hidden_size

        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.input_size = self.lstm_hidden *2

        self.hidden = nn.Linear(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input):
        """
        inputs: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """

        metnion_words, metnion_lengths, metnion_seq_recover = input
        metnion_words_embeds = self.embedding(metnion_words)
        batch_size, metnion_max_len, _ = metnion_words_embeds.size()
        mention_packed_words = pack_padded_sequence(metnion_words_embeds, metnion_lengths.cpu().numpy(), True)
        mention_hidden = None
        mention_lstm_out, mention_hidden = self.lstm(mention_packed_words, mention_hidden)
        mention_lstm_out, _ = pad_packed_sequence(mention_lstm_out)

        hid_size = mention_lstm_out.size(2) // 2
        mention_bilstm_out = torch.cat([mention_lstm_out[0, :, :hid_size], mention_lstm_out[-1, :, hid_size:]],dim=1)  # metnion bilstm output

        #or get mention_lstm_out
        # mention_lstm_out_2 = mention_lstm_out.transpose(1, 0)
        # mention_bilstm_out = torch.cat([mention_lstm_out_2[:, 0, :hid_size], mention_lstm_out_2[:, -1, hid_size:]],dim=1)  # metnion bilstm output

        mention_bilstm_out = self.dropout(mention_bilstm_out)

        # sent_words, sent_lengths = sents
        # sent_words_embeds = self.embedding(sent_words)
		#
        # packed_words = pack_padded_sequence(sent_words_embeds, sent_lengths.cpu().numpy(), True)
        # sent_hidden = None
        # lstm_out, sent_hidden = self.lstm(packed_words, sent_hidden)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)

        # hid_size = lstm_out.size(2) // 2
        # sents_bilstm_out = torch.cat([lstm_out[0, :, :hid_size], lstm_out[-1, :, hid_size:]],
        #                              dim=1)  # sentence bilstm output
        # feature_out = self.dropout(sents_bilstm_out)
        # inputs_embs = torch.cat((torch.mul(atten_input, 0.9),torch.mul(feature_out, 0.1)), 1)

        hidden = self.hidden(mention_bilstm_out)
        output = self.out(hidden)
        return output


