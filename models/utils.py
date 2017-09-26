import torch

def initialize_embs(params, vocab):
    # don't train the word embeddings
    found_embeddings = False

    print "Initializing embeddings..."

    for param in params():
        if param.size()[0] == vocab.size() + 1:
            found_embeddings = True
            # keep default randomly initialized value for the UNK token
            param.data[:-1] = torch.FloatTensor(vocab.idx2Emb[:-1])
            param.requires_grad = False

    if found_embeddings:
        print "Done initializing embeddings..."
    else:
        raise Exception("Couldn't find word embeddings.\
        Make sure parameter size is 'vocab.size() + 1'")
