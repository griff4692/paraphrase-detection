import argparse
import json
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim

from vocab import Vocab
from batcher import Batcher

from models.CBOW import CBOW
from models.Tree import Tree

def get_model(args, vocab):
    if args.model_name == 'CBOW':
        args.test_batch_size = 128
        return CBOW(vocab, args.embed_dim)
    elif args.model_name == 'TREE':
        args.test_batch_size = 1
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

def test(model, test_batcher, vocab, args):
    test_data = resolve_data(args, "test")

    num_tested = 0
    num_wrong = 0
    true_positives = 0
    true_predicted_positives = 0
    num_positive_labels = 0
    while not test_batcher.is_finished():
        sentences1, sentences2, labels = test_batcher.get_batch()
        x, y = model.prepare_batch(sentences1, sentences2, labels)

        num_positive_labels += sum(labels)

        predictions = model(x).data.numpy()

        predictions = np.reshape(predictions, (test_batcher.batch_size,))
        # simulate sigmoid + prediction
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0

        label_pre_sum = predictions + labels
        true_positives = (label_pre_sum == 2).shape[0]

        num_predicted_positives += np.sum(predictions)

        abs_deltas = np.abs(predictions - labels)

        num_wrong += np.sum(abs_deltas)
        num_tested += predictions.shape[0]

    precision = float(true_positives) / predicted_positives
    recall = float(true_positives) / num_positive_labels

    F_score = 2.0 / ((1.0 / recall) + (1.0 / precision))

    return float(num_tested - num_wrong) / num_tested, F_score

def resolve_vocab(args):
    # build up vocabulary
    embed_path = './embeddings/glove.6B/glove.6B.%dd.txt' % (args.embed_dim)
    vocab = Vocab()
    if args.build_vocab_from_sources:
        print "Loading stored vocab..."
        vocab.load(args.embed_dim)
        print "Done loading stored vocab..."
    else:
        vocab.build(train_data[:, 3:5], embed_path, args.embed_dim)

    return vocab

def resolve_data(args, flavor = 'train'):
    if args.use_preprocessed:
        with open('./data/msr_paraphrase_' + flavor + '.json', 'rb') as data_reader:
            data = json.load(data_reader)
    else:
        data_reader = open('./data/msr_paraphrase_' + flavor + '.txt', 'rb')
        data = np.array([example.split("\t") for example in data_reader.readlines()])[1:]

    return data

def train(args):
    # retrieve proper data, model, and vocabulary
    train_data = resolve_data(args, "train")
    test_data = resolve_data(args, "test")
    vocab = resolve_vocab(args)
    model = get_model(args, vocab)

    # intialize batchers
    train_batcher = Batcher(train_data, args.batch_size, args.use_preprocessed)
    test_batcher = Batcher(test_data, args.test_batch_size, args.use_preprocessed)

    # initialize training parameters
    loss = torch.nn.BCEWithLogitsLoss()
    # don't optimizer fixed weights like GloVe embeddings
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # evaluation metrics
    best_accuracy = 0.0
    best_params = None
    best_epoch = 0
    prev_accuracy = 0
    consec_worse_epochs = 0

    for i in range(args.epochs):
        cost = 0.
        while not train_batcher.is_finished():
            sentences1, sentences2, labels = train_batcher.get_batch()
            cost += train_batch(model, loss, optimizer, sentences1, sentences2, labels, vocab)

        print("Epoch = %d, average loss = %s" % (i + 1, cost / train_batcher.num_batches))

        if (i + 1) % args.test_freq == 0:
            test_acc, F_score = test(model, test_batcher, vocab, args)
            print("Accuracy (F-score) after epoch #%s --> %s%% (%s)" % (i, int(acc * 100.0), F_score))

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
    acc, F_score = test(model, test_batcher, vocab, args)
    print("Best Accuracy achieved after epoch #%s --> %s%% (%s" % (best_epoch, int(acc * 100.0), F_score))

def main():
    parser = argparse.ArgumentParser(description='Paraphrase Detection Training Parameters.')
    parser.add_argument('--model_name', default='TREE')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--max_consec_worse_epochs', type=int, default=3)
    parser.add_argument('--build_vocab_from_sources', type=int, default=1) # 0 False, True is 1
    parser.add_argument('--use_preprocessed', type=int, default=1) # 0 False, 1 True
    parser.add_argument('--test_freq', type=int, default=5)

    args = parser.parse_args()

    if(args.model_name == 'TREE' and not args.batch_size == 1):
        print "\nWARNING -->TREE Models can only be trained with batch size 1\n"
        print "...switching to batch size 1"
        args.batch_size = 1

    train(args)

if __name__ == "__main__":
    main()
