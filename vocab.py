import numpy as np

class Vocab:
    def __init__(self):
        self.word2Idx = {}
        self.idx2Word = []
        self.idx2Emb = []

    def tokenizer(self, word_str):
        return [word.lower() for word in word_str.strip().split()]

    def index_tokenizer(self, word_str):
        return [self.word2Idx[word.lower()] if word.lower() in self.word2Idx else self.size() for word in word_str.strip().split()]

    def build(self, sentences, path_to_embed, emb_dims):
        for sentence in sentences:
            tokens_1 = self.tokenizer(sentence[0])
            tokens_2 = self.tokenizer(sentence[1])

            for token in tokens_1 + tokens_2:
                if token not in self.word2Idx:
                    idx = len(self.idx2Word)
                    self.idx2Word.append(token)
                    self.word2Idx[token] = idx

        self.build_emb_matrix(path_to_embed, emb_dims)

    def size(self):
        return len(self.idx2Word)

    def build_emb_matrix(self, path_to_embed, emb_dims):
        self.emb_dims = emb_dims
        self.idx2Emb = np.zeros((self.size() + 1, emb_dims), dtype='float')

        embeddings = open(path_to_embed)
        for embedding in embeddings:
            split = embedding.split()
            word = split[0]

            if word in self.word2Idx:
                vals =  np.array([float(num) for num in split[1:]])
                self.idx2Emb[self.word2Idx[word]] = vals
