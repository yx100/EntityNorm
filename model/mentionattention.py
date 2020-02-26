import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
from options import opt

class Attention(nn.Module):
    def __init__(self,vocab, num_classes):
        super(Attention,self).__init__()
        self.embedding = vocab.init_embed_layer()
        self.hidden_size = opt.hidden_size
        self.embed_size = self.embedding.weight.size(1)
        self.W = nn.Linear(self.embed_size, 1, bias=False)
        self.hidden = nn.Linear(self.embed_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, num_classes)

        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths, seq_recover = input
        inputs_embeds = self.embedding(inputs) #(N, W,D)
        batch_size, max_len,_ = inputs_embeds.size()
        flat_input = inputs_embeds.contiguous().view(-1, self.embed_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda(opt.gpu)
        mask = autograd.Variable((idxes < lengths.unsqueeze(1)).float())

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        atten_input = torch.bmm(alphas.unsqueeze(1), inputs_embeds).squeeze(1)
        atten_input = self.dropout(atten_input)
        hidden = self.hidden(atten_input)

        output = self.out(hidden)
        return output