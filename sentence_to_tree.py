from nltk.parse.stanford import StanfordParser
import os
from queue import Queue
from vocab import Vocab

os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
# os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-english-corenlp-2017-06-09-models.jar'

parser = StanfordParser() # model_path?

def build_tree(raw_sent, vocab):
    tree = list(parser.raw_parse(raw_sent))[0]

    id_counter = 0

    rootNode = TreeNode(tree.label(), tree.height() - 1, id_counter, -1)
    sentence = TreeSentence(rootNode)

    q = Queue()
    q.put((tree, 0, id_counter)) # nltk Tree Object, level, current_id

    while not q.empty():
        curr_subtree, curr_level, curr_id = q.get()

        for subtree in curr_subtree:
            id_counter += 1

            if type(subtree) == unicode:
                print subtree
                print tree
                for sub in tree.subtrees():
                    print sub

                tree.draw()

            subtree_node = TreeNode(subtree.label(), subtree.height() - 1, id_counter, curr_id)
            node_idx = sentence.add_to_level(curr_level + 1, subtree_node)

            if(len(subtree) == 1 and type(subtree[0]) == unicode):
                subtree_node.word = vocab.get(subtree[0].lower())
            else:
                q.put((subtree, curr_level + 1, id_counter))

    return sentence


class TreeSentence:
    def __init__(self, root):
        self.levels = [[] for i in range(root.get_height())]
        self.levels[0].append(root)

    def add_to_level(self, idx, node):
        self.levels[idx].append(node)
        return len(self.levels[idx]) - 1

    def get_height(self):
        return self.levels[0][0].get_height()

    def render(self, vocab=None):
        for (i, level) in enumerate(self.levels):
            print "Level #" + str(i) + "\n"
            for (j, node) in enumerate(level):
                node.pretty_print(self.get_children(i, j), vocab)

            print "\n"

    def get_children(self, level, idx):
        parent_node = self.levels[level][idx]

        if level >= len(self.levels) - 1:
            return []
        else:
            return [
                (i, node) for (i, node) in enumerate(self.levels[level + 1])
                    if node.parent_id == parent_node.my_id
            ]

class TreeNode:
    def __init__(self, label, height, my_id, parent_id):
        self.label = label
        self.word = None
        self.my_id = my_id
        self.height = height
        self.parent_id = parent_id

    def get_height(self):
        return self.height

    def is_leaf(self):
        return self.height() == 1

    def pretty_print(self, child_ids, vocab=None):
        child_ids = [str(child_id[0]) for child_id in child_ids]
        actual_word = '' if self.word is None or vocab is None else vocab.get_word(self.word)
        word_str = '' if self.word is None else 'Word = ' + actual_word + ' (' + str(self.word) + ')'
        print 'ID = ( ' + str(self.my_id) + ' )'
        print '\t--> Label = ' + self.label
        print '\t--> Height = ' + str(self.height)
        print '\t--> Parent ID = ( ' + str(self.parent_id) + ' )'
        print '\t--> Child IDs = [ ' + ", ".join(child_ids) + ' ]'
        if len(word_str) > 0:
            print '\t--> ' + word_str

if __name__== '__main__':
    sentence = "the quick brown fox jumps over the lazy dog"
    vocab = Vocab()
    vocab.load(100)

    tree_sentence = build_tree(sentence, vocab)
    tree_sentence.render(vocab)
