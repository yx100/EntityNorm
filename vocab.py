import numpy as np
import torch
import torch.nn as nn
from options import opt
import logging
import struct

def _readString(f):
    s = str()
    c = f.read(1).decode('iso-8859-1')
    while c != '\n' and c != ' ':
        s = s + c
        c = f.read(1).decode('iso-8859-1')

    return s

def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

class Vocab:

    def __init__(self, alphabet_from_dataset, pretrained_file, emb_size):
        # add UNK with its index 0
        self.unk_tok = '<unk>'
        self.unk_idx = 0
        self.vocab_size = 1
        self.v2wvocab = ['<unk>']
        self.w2vvocab = {'<unk>': 0}

        # add padding with its index 1
        self.pad_tok = '<pad>'
        self.pad_idx = opt.pad_idx
        self.vocab_size += 1
        self.v2wvocab.append('<pad>')
        self.w2vvocab['<pad>'] = self.pad_idx
        # build vocabulary
        self.vocab_size += len(alphabet_from_dataset)
        cnt = 2
        for alpha in alphabet_from_dataset:
            self.v2wvocab.append(alpha)
            self.w2vvocab[alpha] = cnt
            cnt += 1
        # initialize embeddings
        if pretrained_file:
            ct_word_in_pretrained = 0

            if pretrained_file.find('.bin') != -1:
                with open(pretrained_file, 'rb') as f:
                    wordTotal = int(_readString(f))

                    self.emb_size = int(_readString(f))
                    self.embeddings = np.random.uniform(-0.01, 0.01, size=(self.vocab_size, self.emb_size))

                    for i in range(wordTotal):
                        word = _readString(f)

                        word_vector = []
                        for j in range(self.emb_size):
                            word_vector.append(_readFloat(f))
                        word_vector = np.array(word_vector, np.float)

                        f.read(1)  # a line break

                        if word == self.unk_tok or word == self.pad_tok:
                            continue # don't use the pretrained values for unk and pad
                        else:
                            if word in self.w2vvocab: # if we can, use the pretrained value
                                self.embeddings[self.w2vvocab[word]] = word_vector
                                ct_word_in_pretrained += 1

            else:
                with open(pretrained_file, 'r') as inf:
                    parts = inf.readline().split()
                    self.emb_size = int(parts[1]) # use pretrained embedding size
                    self.embeddings = np.random.uniform(-0.01, 0.01, size=(self.vocab_size, self.emb_size))

                    for line in inf.readlines():
                        parts = line.rstrip().split(' ')
                        word = parts[0] # not norm, use original word in the pretrained
                        if word == self.unk_tok or word == self.pad_tok:
                            continue # don't use the pretrained values for unk and pad
                        else:
                            if word in self.w2vvocab: # if we can, use the pretrained value
                                vector = [float(x) for x in parts[-self.emb_size:]]
                                self.embeddings[self.w2vvocab[word]] = vector
                                ct_word_in_pretrained += 1

            print("vocab matched {} in pretrained embeding file".format(100.0*ct_word_in_pretrained/self.vocab_size))

        else:
            self.emb_size = emb_size
            self.embeddings = np.random.uniform(-0.01, 0.01, size=(self.vocab_size, self.emb_size))

        # normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1,1)
        # zero pad emb
        self.embeddings[self.pad_idx] = 0

    def lookup(self, alpha):

        if alpha in self.w2vvocab:
            return self.w2vvocab[alpha]
        return self.unk_idx

    def lookup_id2str(self, id):
        if id<0 or id>=self.vocab_size:
            raise RuntimeError("{}: id out of range".format(self.__class__.__name__))
        return self.v2wvocab[id]

    def init_embed_layer(self):
        word_emb = nn.Embedding(self.vocab_size, self.emb_size, self.pad_idx)
        word_emb.weight.data = torch.FloatTensor(self.embeddings)
        return word_emb

