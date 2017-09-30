import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import *
from torch.autograd import Variable
from sentence_to_tree import build_tree, TreeSentence

class Tree(nn.Module):
    def __init__(self, vocab, embed_dim):
        super(Tree, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab.size() + 1, embed_dim)

        self.affine_transform = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.affine_bias = nn.Parameter(torch.zeros(self.embed_dim))

        self.pool_target_size = 10
        self.pool_window_size = 3
        self.dynamic_max_pool = nn.FractionalMaxPool2d(self.pool_window_size,
            output_size=(self.pool_target_size, self.pool_target_size))
        self.output = nn.Linear(self.pool_target_size * self.pool_target_size, 1)
        initialize_embs(self.parameters, self.vocab)

    def prepare_batch(self, batch_sentences1, batch_sentences2, labels):
        labels = Variable(torch.FloatTensor(labels), requires_grad=False)
        return (batch_sentences1, batch_sentences2), labels


    # Flatten according to paper
    # first by original word-level sentence order (tree.leaves)
    def flatten(self, matrix, tree, non_terminal_ct):
        leaf_embeds = self.embeddings(torch.LongTensor(tree.leaves))
        leaf_len = len(tree.leaves)
        idx = leaf_len

        flattened = Variable(torch.zeros(leaf_len + non_terminal_ct, self.embed_dim))

        flattened[:leaf_len] = leaf_embeds

        for r in range(len(matrix) - 1, -1, -1):
            row = matrix[r]
            for c in range(len(row) - 1, -1, -1):
                if not tree.levels[r + 1][c].is_leaf():
                    flattened[idx, :] = row[c]
                    idx += 1

        return flattened

    def forward(self, (batch_trees1, batch_trees2)):
        batch_size = batch_trees1.shape[0]
        batch_flat_sim_mats = Variable(torch.zeros([batch_size, self.pool_target_size * self.pool_target_size ]))

        for batch_idx in range(batch_size):
            tree1 = batch_trees1[batch_idx][0]
            tree2 = batch_trees2[batch_idx][0]

            trees = [
                TreeSentence(**tree1),
                TreeSentence(**tree2)
            ]

            num_subtrees = [0, 0]
            hidden_states = []

            for (no, tree) in enumerate(trees):
                hidden_states.append([])

                for i in range(len(tree.levels) - 1): # no need to include root node
                    hidden_states[no].append([])

                for r in range(len(tree.levels) - 1, 0, -1): # don't include root node
                    level = tree.levels[r]
                    for (c, node) in enumerate(level):
                        if node.is_leaf():
                            # if compound word like [1, 5/8] - take mean of each components embedding
                            h = self.embeddings(torch.LongTensor([node.token]))
                            hidden_states[no][r - 1].append(h)
                        else:
                            num_subtrees[no] += 1
                            children = tree.get_children(r, c)
                            for i in range(len(children)):
                                curr_h = hidden_states[no][r][children[i][0]]
                                if i == 0:
                                    new_h = self.affine_transform(curr_h)
                                else:
                                    new_h += self.affine_transform(curr_h)

                            output_h = F.tanh(new_h + self.affine_bias)
                            hidden_states[no][r - 1].append(output_h)

            # Column-wise tensor of subtree hidden states
            sent_1_mat =  self.flatten(hidden_states[0], trees[0], num_subtrees[0]) # |subtrees1|, embed
            sent_2_mat =  self.flatten(hidden_states[1], trees[1], num_subtrees[1]) # |subtrees2|, embed

            # Compute similarity matrix
            sent_1_mat_stacked = torch.stack([sent_1_mat] * sent_2_mat.size()[0]) # |s1|, |s2|, embed
            sent_2_mat_stacked = torch.stack([sent_2_mat] * sent_1_mat.size()[0]).transpose(0, 1) # |s1|, |s2|, embed

            sim_mat = torch.sum((sent_1_mat_stacked - sent_2_mat_stacked)**2, 2).unsqueeze(0)

            sim_mat_pool = self.dynamic_max_pool(sim_mat)
            sim_mat_flat = sim_mat_pool.view(1, self.pool_target_size * self.pool_target_size)

            std = sim_mat_flat.std()
            mean = sim_mat_flat.mean()
            sim_mat_flat_norm = (sim_mat_flat - mean) / std

            batch_flat_sim_mats[batch_idx] = sim_mat_flat_norm

        return self.output(batch_flat_sim_mats)
