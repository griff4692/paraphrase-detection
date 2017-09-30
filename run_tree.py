from models.CBOW import CBOW
from models.Tree import Tree
import util
import argparse
import eval

def get_model(args, vocab):
    if args.model_name == 'CBOW':
        return CBOW(vocab, args.embed_dim)
    elif args.model_name == 'TREE':
        return Tree(vocab, args.embed_dim)
    else:
        raise Exception("Unsupported Model name --> %s" % (args.model_name))

def main():
    parser = argparse.ArgumentParser(description='Paraphrase Detection Training Parameters.')
    parser.add_argument('--model_name', default='TREE')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001, description='Initial learning rate to pass to optimizer.')
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--max_consec_worse_epochs', type=int, default=3)
    parser.add_argument('--build_vocab_from_sources', type=int, default=1) # 0 False, True is 1
    parser.add_argument('--use_preprocessed', type=int, default=1) # 0 False, 1 True
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=128)

    args = parser.parse_args()
    train_data = util.resolve_data(args, "train")
    test_data = util.resolve_data(args, "test")
    vocab = util.resolve_vocab(args, train_data)
    model = get_model(args, vocab)
    eval.train(args, model, train_data, test_data, vocab)

if __name__ == "__main__":
    main()