from nltk.parse.stanford import StanfordParser
import os
from queue import Queue
from vocab import Vocab

os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
# os.environ['STANFORD_MODELS'] = '/Users/griffinadams/Desktop/stanford-english-corenlp-2017-06-09-models.jar'

parser = StanfordParser() # model_path?


def glob_compound_leaves(leaves, word):
    compounds = word.split('-')

    if len(compounds) == 1:
        return leaves

    if len(compounds) == 3:
        raise Exception("Triple Compound= " + word)

    for (i, leaf) in enumerate(leaves):
        if compounds[0] == leaf and compounds[1] == leaves[i + 1]:
            # both are equal to compound
            leaves[i] = word
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
                word = subtree[0]
                if len(subtree) > 1:
                    word = ''
                    for (i, subsubtree) in enumerate(subtree):
                        if i > 0:
                            word += '-'
                        word += subsubtree.lower()
                        leaves = glob_compound_leaves(leaves, word)

                subtree_node.word = word
                subtree_node.token = vocab.get(word)
            else:
                q.put((subtree, curr_level + 1, id_counter))

    sentence.add_leaves(leaves)
    return sentence


class TreeSentence:
    def __init__ (self, root_or_levels, leaves = []):
        self.leaves = leaves
        if isinstance(root_or_levels, TreeNode):
            root = root_or_levels
            self.levels = [[] for i in range(root.get_height())]
            self.levels[0].append(root)
        else:
            levels = root_or_levels
            self.levels = levels
            for (r, level) in enumerate(levels):
                for(c, node_dict) in enumerate(level):
                    if(not isinstance(node_dict, TreeNode)):
                        self.levels[r][c] = TreeNode(**node_dict)

    def add_to_level(self, idx, node):
        self.levels[idx].append(node)
        return len(self.levels[idx]) - 1

    def leaf_idxs(self, word):
        return [idx for idx, leaf_word in enumerate(leaves) if word == leaf_word]

    def add_leaves(self, leaves):
        self.leaves = leaves

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
    sentence = "The 1 5/8 percent note maturing in April 2005 gained 1/16 to 100 13/32, lowering its yield 1 basis points to 1.41 percent.".decode('utf-8')
    vocab = Vocab()
    vocab.load(100)
    tree_sentence = build_tree(sentence, vocab)
    tree_sentence.render(vocab)
