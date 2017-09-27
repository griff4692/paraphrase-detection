import os
import json
import numpy as np
from vocab import Vocab
from sentence_to_tree import build_tree

os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'

BUILD_VOCAB_FROM_SOURCES = True # switch to False if you want to redo vocab ugh
EMBED_DIM = 100
EMBED_PATH = './embeddings/glove.6B/glove.6B.%dd.txt' % (EMBED_DIM)

if __name__ == '__main__':
    train_reader =  open('./data/msr_paraphrase_train.txt', 'rb')
    unprocessed_train_data = np.array([example.split("\t") for example in train_reader.readlines()])[1:]

    # build up vocabulary
    vocab = Vocab()

    if BUILD_VOCAB_FROM_SOURCES:
        vocab.load(EMBED_DIM)
    else:
        vocab.build(unprocessed_train_data[:,3:5], EMBED_PATH, EMBED_DIM)
        vocab.save()

    print "Size of Vocab = " + str(vocab.size())

    flavors = ['train', 'test']

    for flavor in flavors:
        reader =  open('./data/msr_paraphrase_' + flavor + '.txt', 'rb')
        unprocessed_data = np.array([example.split("\t") for example in reader.readlines()])[1:]

        data = []

        for (i, example) in enumerate(unprocessed_data):
            new_example = []
            new_example.append(float(example[0]))

            raw_sentence_1 = example[3]
            raw_sentence_2 = example[4]

            tokens_1 = vocab.tokenizer(raw_sentence_1)
            tokens_2 = vocab.tokenizer(raw_sentence_2)

            idxs_1 = vocab.tokens_to_idxs(tokens_1)
            idxs_2 = vocab.tokens_to_idxs(tokens_2)

            tree_1 = build_tree(raw_sentence_1.decode('utf-8'), vocab)
            tree_2 = build_tree(raw_sentence_2.decode('utf-8'), vocab)

            new_example.append(raw_sentence_1)
            new_example.append(tokens_1)
            new_example.append(idxs_1)
            new_example.append(tree_1)

            new_example.append(raw_sentence_2)
            new_example.append(tokens_2)
            new_example.append(idxs_2)
            new_example.append(tree_2)

            data.append(new_example)

            if((i + 1) % 10 == 0):
                print("Processing " + str(i + 1) + " / " + str(unprocessed_data.shape[0]) + " examples for " + flavor)

        with open('data/msr_paraphrase_' + flavor + '.json', 'wb') as outfile:
            json.dump(data, outfile, default=lambda o: o.__dict__)
