import numpy as np
import torch
from torch.autograd import Variable
from torch import optim

from batcher import Batcher

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
    num_tested = 0
    num_wrong = 0
    true_positives = 0
    num_predicted_positives = 0
    num_positive_labels = 0
    while not test_batcher.is_finished():
        sentences1, sentences2, labels = test_batcher.get_batch()
        x, y = model.prepare_batch(sentences1, sentences2, labels)

        num_positive_labels += np.sum(labels)

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

    precision = float(true_positives) / num_predicted_positives
    recall = float(true_positives) / num_positive_labels

    F_score = 2.0 / ((1.0 / recall) + (1.0 / precision))

    return float(num_tested - num_wrong) / num_tested, F_score

def train(args, model, train_data, test_data, vocab):
    # retrieve proper data, model, and vocabulary

    # intialize batchers
    train_batcher = Batcher(train_data, args.batch_size, args.model_name)
    test_batcher = Batcher(test_data, args.test_batch_size, args.model_name)

    # initialize training parameters
    loss = torch.nn.BCEWithLogitsLoss()
    # don't optimizer fixed weights like GloVe embeddings
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

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
                    print("Training incurred %s consecutive worsening epoch(s): from %s to %s"
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

