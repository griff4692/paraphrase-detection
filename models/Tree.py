import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import *
from sentence_to_tree import Sentence2Tree

class Tree(nn.Module):
    def __init__(self, vocab, embed_dim):
        super(Tree, self).__init__()
        self.vocab = vocab
        self.embeddings = nn.Embedding(vocab.size() + 1, embed_dim)
        initialize_embs(self.parameters, self.vocab)

        # don't train the word embeddings
        for param in self.parameters():
            if param.size()[0] == self.vocab.size() + 1:
                # keep default randomly initialized value for the UNK token
                param.data[:-1] = torch.FloatTensor(vocab.idx2Emb[:-1])
                param.requires_grad = False

    def prepare_batch(self, batch_sentence1, batch_sentence2, label):
        tree1 = Sentence2Tree.build_tree(batch_sentence1)
        tree2 = Sentence2Tree.build_tree(batch_sentence2)

        label = np.expand_dims(np.array(label, dtype='float'), axis=-1)
        label = Variable(torch.FloatTensor(label), requires_grad=False)

        return (tree1, tree2), label

    def forward(self, (tree1, tree2)):
        for level in tree1.levels:
            print level
