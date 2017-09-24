import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class CBOW(nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dims = 128):
        super(CBOW, self).__init__()
        self.vocab = vocab
        self.embeddings = nn.EmbeddingBag(vocab.size() + 1, embedding_dim, mode='sum')
        self.linear1 = nn.Linear(embedding_dim * 2, hidden_dims)
        self.output = nn.Linear(hidden_dims, 1)


    def prepare_batch(self, batch_sentences1, batch_sentences2, labels):
        batch_size = len(batch_sentences1)
        assert batch_size == len(batch_sentences2)

        offsets1 = [0] * batch_size
        offsets2 = [0] * batch_size

        bow1 = []
        bow2 = []

        for i in range(batch_size):
            tokens1 = self.vocab.index_tokenizer(batch_sentences1[i])
            tokens1_len = len(tokens1)

            tokens2 = self.vocab.index_tokenizer(batch_sentences2[i])
            tokens2_len = len(tokens2)

            bow1 += tokens1
            bow2 += tokens2

            if(len(tokens1) == 0 or len(tokens2) == 0):
                raise

            if i > 0:
                offsets1[i] = offsets1[i - 1] + tokens1_len
                offsets2[i] = offsets2[i - 1] + tokens2_len

        bow1 = Variable(torch.LongTensor(bow1), requires_grad=False)
        bow2 = Variable(torch.LongTensor(bow2), requires_grad=False)

        offsets1 = Variable(torch.LongTensor(offsets1), requires_grad=False)
        offsets2 = Variable(torch.LongTensor(offsets2), requires_grad=False)

        labels = np.expand_dims(np.array(labels, dtype='float'), axis=-1)
        labels = Variable(torch.FloatTensor(labels), requires_grad=False)

        return bow1, offsets1, bow2, offsets2, labels

    def forward(self, bow1, offsets1, bow2, offsets2):
        cbow1 = self.embeddings(bow1, offsets1)
        cbow2 = self.embeddings(bow2, offsets2)

        cbow_concat = torch.cat((cbow1, cbow2), 1)

        fc1 = F.relu(self.linear1(cbow_concat))
        return self.output(fc1)
