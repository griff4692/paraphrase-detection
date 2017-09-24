import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim
from vocab import Vocab
from CBOW import CBOW

BATCH_SIZE = 32
EMBED_DIM = 100
EPOCHS = 100
MAX_CONSEC_WORSE_EPOCHS = 1
TEST_FREQ = 5

def train(model, loss, optimizer, sentences1, sentences2, labels, vocab):
    bow1, offsets1, bow2, offsets2, labels = model.prepare_batch(sentences1, sentences2, labels)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model(bow1, offsets1, bow2, offsets2)
    output = loss.forward(fx, labels)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]

def test(model, vocab):
    test_reader = open('./data/msr_paraphrase_test.txt', 'rb')
    test_data = np.array([example.split("\t") for example in test_reader.readlines()][1:])

    num_batches = len(test_data) // BATCH_SIZE
    num_tested = 0
    num_wrong = 0
    for k in range(num_batches):
        start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
        sentences1 = test_data[start:end, 3]
        sentences2 = test_data[start:end, 4]
        labels = np.array(test_data[start:end, 0], dtype='float')
        bow1, offsets1, bow2, offsets2, _ = model.prepare_batch(sentences1, sentences2, labels)
        predictions = model(bow1, offsets1, bow2, offsets2).data.numpy()

        predictions = np.reshape(predictions, (BATCH_SIZE,))
        # simulate sigmoid + prediction
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0

        abs_deltas = np.abs(predictions - labels)

        num_wrong += np.sum(abs_deltas)
        num_tested += predictions.shape[0]

    return float(num_tested - num_wrong) / num_tested


def main():
    train_reader = open('./data/msr_paraphrase_train.txt', 'rb')
    train_data = np.array([example.split("\t") for example in train_reader.readlines()][1:])

    vocab = Vocab()
    embed_path = './embeddings/glove.6B/glove.6B.%dd.txt' % (EMBED_DIM)
    vocab.build(train_data[:, 3:5], embed_path, EMBED_DIM)

    model = CBOW(vocab, EMBED_DIM)

    for param in model.parameters():
        if param.size()[0] == vocab.size() + 1:
            param.data = torch.FloatTensor(vocab.idx2Emb)
            param.requires_grad = False

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    best_accuracy = 0.0
    best_params = None
    best_epoch = 0
    prev_accuracy = 0
    consec_worse_epochs = 0
    for i in range(EPOCHS):
        cost = 0.
        num_batches = len(train_data) // BATCH_SIZE
        for k in range(num_batches):
            start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
            sentences1 = train_data[start:end, 3]
            sentences2 = train_data[start:end, 4]
            labels = train_data[start:end, 0]
            cost += train(model, loss, optimizer, sentences1, sentences2, labels, vocab)

        print("Epoch = %d, average loss = %s" % (i + 1, cost / num_batches))
        np.random.shuffle(train_data)

        if (i + 1) % TEST_FREQ == 0:
            test_acc = test(model, vocab)

            if test_acc < prev_accuracy:
                consec_worse_epochs += 1
                if consec_worse_epochs >= MAX_CONSEC_WORSE_EPOCHS:
                    print("Training incurred %s consecutive worsening epoch(s): from %s to %s" \
                    % (MAX_CONSEC_WORSE_EPOCHS, i + 1 - (MAX_CONSEC_WORSE_EPOCHS * TEST_FREQ), i + 1))
                    break
            else:
                consec_worse_epochs = 0

                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_epoch = i + 1
                    best_params = model.state_dict()

            prev_accuracy = test_acc

    model.load_state_dict(best_params)
    acc = test(model, vocab)
    print("Best Accuracy achieved after epoch #%s --> %s%%" % (best_epoch, int(acc * 100.0)))

if __name__ == "__main__":
    main()
