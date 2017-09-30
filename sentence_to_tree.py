from nltk.parse.stanford import StanfordParser
import os
from queue import Queue
from vocab import Vocab

os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
# os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-english-corenlp-2017-06-09-models.jar'

parser = StanfordParser() # model_path?


def glob_compound_leaves(leaves, compound1, compound2 = None):
    if compound2 is None:
        return leaves

    for (i, leaf) in enumerate(leaves):
        if compound1 == leaf and compound2 == leaves[i + 1]:
            # both are equal to compound
            leaves[i] = compound1 + "-" + compound2
            leaves = leaves[:i + 1] + leaves[i + 2:]
            return leaves

    raise Exception("Should be able to find compount leaf words!")


def build_tree(raw_sent, vocab):
    tree = list(parser.raw_parse(raw_sent))[0]

    leaves = [leaf.lower() for leaf in tree.leaves()]

    id_counter = 0

    rootNode = TreeNode(tree.label(), tree.height() - 1, id_counter, -1, None, None)
    sentence = TreeSentence(rootNode)

    q = Queue()
    q.put((tree, 0, id_counter)) # nltk Tree Object, level, current_id

    while not q.empty():
        curr_subtree, curr_level, curr_id = q.get()

        for subtree in curr_subtree:
            id_counter += 1

            subtree_node = TreeNode(subtree.label(), subtree.height() - 1, id_counter, curr_id, None, None)
            node_idx = sentence.add_to_level(curr_level + 1, subtree_node)

            if type(subtree[0]) == unicode:
                word = subtree[0].lower()
                if len(subtree) > 1:
                    word2 = subtree[1].lower()
                    leaves = glob_compound_leaves(leaves, word, word2)
                    word += "-" + word2

                subtree_node.word = word
                subtree_node.token = vocab.get(word)
            else:
                q.put((subtree, curr_level + 1, id_counter))

    sentence.add_leaves(leaves, vocab)
    return sentence


class TreeSentence:
    def __init__ (self, levels, leaves = []):
        self.leaves = leaves
        if isinstance(levels, TreeNode):
            root = levels
            self.levels = [[] for i in range(root.get_height())]
            self.levels[0].append(root)
        else:
            self.levels = levels
            for (r, level) in enumerate(levels):
                for(c, node_dict) in enumerate(level):
                    if(not isinstance(node_dict, TreeNode)):
                        self.levels[r][c] = TreeNode(**node_dict)

    def add_to_level(self, idx, node):
        self.levels[idx].append(node)
        return len(self.levels[idx]) - 1

    def add_leaves(self, leaves, vocab):
        self.leaves = [vocab.get(leaf) for leaf in leaves]

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
    def __init__(self, label, height, my_id, parent_id, token, word):
        self.label = label
        self.token = token
        self.word = word
        self.my_id = my_id
        self.height = height
        self.parent_id = parent_id

    def get_height(self):
        return self.height

    def is_leaf(self):
        return self.get_height() == 1

    def pretty_print(self, child_ids, vocab=None):
        child_ids = [str(child_id[0]) for child_id in child_ids]
        print 'ID = ( ' + str(self.my_id) + ' )'
        print '\t--> Label = ' + self.label
        print '\t--> Height = ' + str(self.height)
        print '\t--> Parent ID = ( ' + str(self.parent_id) + ' )'
        print '\t--> Child IDs = [ ' + ", ".join(child_ids) + ' ]'
        if self.word is not None:
            print '\t--> Word = ' + self.word + ' (' + str(self.token) + ')'

if __name__== '__main__':
    sentence = "the cats from new york catch mice".decode('utf-8')
    vocab = Vocab()
    vocab.load(100)
    tree_sentence = build_tree(sentence, vocab)
    tree_sentence.render(vocab)
