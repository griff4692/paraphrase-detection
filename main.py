import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim
import vocab
from models.CBOW import CBOW
from models.Tree import Tree

def get_model(args, vocab):
    if args.model_name == 'CBOW':
        return CBOW(vocab, args.embed_dim)
    elif args.model_name == 'TREE':
        return Tree(vocab, args.embed_dim)
    else:
        raise Exception("Unsupported Model name --> %s" % (args.model_name))

def train_batch(model, loss, optimizer, sentences1, sentences2, labels, vocab):
    x, y = model.prepare_batch(sentences1, sentences2, labels)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]

def test(model, vocab):
    TEST_BATCH_SIZE = 128
    test_reader = open('./data/msr_paraphrase_test.txt', 'rb')
    test_data = np.array([example.split("\t") for example in test_reader.readlines()][1:])

    num_batches = len(test_data) // TEST_BATCH_SIZE
    num_tested = 0
    num_wrong = 0
    for k in range(num_batches):
        start, end = k * TEST_BATCH_SIZE, (k + 1) * TEST_BATCH_SIZE
        sentences1 = test_data[start:end, 3]
        sentences2 = test_data[start:end, 4]
        labels = np.array(test_data[start:end, 0], dtype='float')
        bow1, offsets1, bow2, offsets2, _ = model.prepare_batch(sentences1, sentences2, labels)
        predictions = model(bow1, offsets1, bow2, offsets2).data.numpy()

        predictions = np.reshape(predictions, (TEST_BATCH_SIZE,))
        # simulate sigmoid + prediction
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0

        abs_deltas = np.abs(predictions - labels)

        num_wrong += np.sum(abs_deltas)
        num_tested += predictions.shape[0]

    return float(num_tested - num_wrong) / num_tested

def train(args):
    train_reader = open('./data/msr_paraphrase_train.txt', 'rb')
    train_data = np.array([example.split("\t") for example in train_reader.readlines()])[1:]

    # build up vocabulary
    embed_path = './embeddings/glove.6B/glove.6B.%dd.txt' % (args.embed_dim)
    vocab = Vocab()
    if args.build_vocab_from_sources:
        print "Loading stored vocab..."
        vocab.load(args.embed_dim)
        print "Done loading stored vocab..."
    else:
        vocab.build(train_data[:,3:5], embed_path, args.embed_dim)

    model = get_model(args, vocab)

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    best_accuracy = 0.0
    best_params = None
    best_epoch = 0
    prev_accuracy = 0
    consec_worse_epochs = 0
    for i in range(args.epochs):
        cost = 0.
        num_batches = len(train_data) // args.batch_size
        for k in range(num_batches):
            start, end = k * args.batch_size, (k + 1) * args.batch_size
            sentences1 = train_data[start:end, 3]
            sentences2 = train_data[start:end, 4]
            labels = train_data[start:end, 0]
            cost += train_batch(model, loss, optimizer, sentences1, sentences2, labels, vocab)

        print("Epoch = %d, average loss = %s" % (i + 1, cost / num_batches))
        np.random.shuffle(train_data)

        if (i + 1) % args.test_freq == 0:
            test_acc = test(model, vocab)

            if test_acc < prev_accuracy:
                consec_worse_epochs += 1
                if consec_worse_epochs >= args.max_consec_worse_epochs:
                    print("Training incurred %s consecutive worsening epoch(s): from %s to %s" \
                    % (args.max_consec_worse_epochs, i + 1 - (args.max_consec_worse_epochs * args.test_freq), i + 1))
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

def main():
    parser = argparse.ArgumentParser(description='Paraphrase Detection Training Parameters.')
    parser.add_argument('--model_name', default='TREE')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--max_consec_worse_epochs', type=int, default=3)
    parser.add_argument('--build_vocab_from_sources', type=int, default=1) # 0 False, True is 1
    parser.add_argument('--test_freq', type=int, default=5)

    args = parser.parse_args()

    if(args.model_name == 'TREE' and not args.batch_size == 1):
        print "\nWARNING -->TREE Models can only be trained with batch size 1\n"
        print "...switching to batch size 1"
        args.batch_size = 1

    train(args)

if __name__ == "__main__":
    main()
