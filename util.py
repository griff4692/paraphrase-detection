import json
import numpy as np
from vocab import Vocab

def resolve_data(args, flavor):
    if args.use_preprocessed:
        with open('./data/msr_paraphrase_' + flavor + '.json', 'rb') as data_reader:
            data = np.array(json.load(data_reader))
    else:
        data_reader = open('./data/msr_paraphrase_' + flavor + '.txt', 'rb')
        data = np.array([example.split("\t") for example in data_reader.readlines()])[1:]
    return data


def resolve_vocab(args, train_data):
    # build up vocabulary
    embed_path = './embeddings/glove.6B/glove.6B.%dd.txt' % (args.embed_dim)
    vocab = Vocab()
    if args.build_vocab_from_sources == 1:
        print "Loading stored vocab..."
        vocab.load(args.embed_dim)
        print "Done loading stored vocab..." + str(vocab.size()) + ' words'
    else:
        vocab.build(train_data[:, 3:5], embed_path, args.embed_dim)

    return vocab