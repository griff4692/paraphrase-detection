import numpy as np
from nltk.parse.stanford import StanfordParser
import os
import pickle

os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'

UNK = "<UNK>"


class Vocab:
    def __init__(self):
        self.word2Idx = {}
        self.idx2Word = []
        self.idx2Emb = []
        self.parser = StanfordParser()

    def load(self, embed_size):
        self.word2Idx = pickle.load(open('vocab/word2Idx.pk', 'rb'))
        self.idx2Word = pickle.load(open('vocab/idx2Word.pk', 'rb'))
        self.idx2Emb = pickle.load(open('vocab/idx2Emb_%d.pk' % (embed_size), 'rb'))

    def save(self):
        pickle.dump(self.word2Idx, open('vocab/word2Idx.pk', 'wb'))
        pickle.dump(self.idx2Word, open('vocab/idx2Word.pk', 'wb'))
        pickle.dump(self.idx2Emb, open('vocab/idx2Emb_%d.pk' % (self.emb_dim), 'wb'))


    def is_fraction(self, string):
        split = string.split('/')

        return len(split) == 2 and split[0].isdigit() and split[1].isdigit()

    def tokenizer(self, word_str):
        word_str = word_str.decode('utf-8')
        parsed_sentence = list(self.parser.raw_parse(word_str))[0].leaves()

        tokens = []
        for (i, token) in enumerate(parsed_sentence):
            token = token.lower()
            if token.isdigit() and i < len(parsed_sentence) - 1:
                if self.is_fraction(parsed_sentence[i + 1]):
                    fraction = token + '-' + parsed_sentence[i + 1]
                    tokens.append(token + '-' + parsed_sentence[i + 1])
                else:
                    tokens.append(token)
            else:
                tokens.append(token)

        return tokens

    def get(self, token):
        return self.word2Idx[token] if token in self.word2Idx else self.unk_idx()

    def get_word(self, idx):
        if idx < 0 or idx >= self.size():
            return UNK
        else:
            return self.idx2Word[idx]

    def tokens_to_idxs(self, tokens):
        return [self.get(token) for token in tokens]

    def sentence_to_idxs(self, sentence_str):
        tokens = self.tokenizer(sentence_str)
        return self.tokens_to_idxs(tokens)

    def build(self, sentences, path_to_embed, emb_dim):
        self.emb_dim = emb_dim
        self.max_sent_length = 0
        for sentence in sentences:
            tokens_1 = self.tokenizer(sentence[0])
            tokens_2 = self.tokenizer(sentence[1])

            len1 = len(tokens_1)
            len2 = len(tokens_2)

            self.max_sent_length = np.max([len1, len2, self.max_sent_length])

            for token in tokens_1 + tokens_2:
                if token not in self.word2Idx:
                    idx = len(self.idx2Word)
                    self.idx2Word.append(token)
                    self.word2Idx[token] = idx

        self.build_emb_matrix(path_to_embed, emb_dim)

    def unk_idx(self):
        return self.size()

    def size(self):
        return len(self.idx2Word)

    def build_emb_matrix(self, path_to_embed, emb_dim):
        self.emb_dim = emb_dim
        self.idx2Emb = np.zeros((self.size() + 1, emb_dim), dtype='float')

        hits = 0

        embeddings = open(path_to_embed)
        for embedding in embeddings:
            split = embedding.split()
            word = split[0]

            if word in self.word2Idx:
                vals =  np.array([float(num) for num in split[1:]])
                self.idx2Emb[self.word2Idx[word]] = vals
                hits += 1

        print("%s out of %s word embeddings initialized." % (hits, self.size()))
