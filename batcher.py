import numpy as np

class Batcher:
    def __init__(self, data, batch_size, model_name):
        self.model_name = model_name

        # each data row is
        # 0 - label
        # 1 - sent1_raw, 2 - sent1_tokens, 3 - sent1_idxs, 4 - sent1_tree,
        # 5 - sent2_raw, 6 - sent2_tokens, 7 - sent2_idxs, 8- sent2_tree ]
        
        self.sentence1_idx = 4 if self.model_name == "TREE" else 3
        self.data = data
        self.batch_size = batch_size
        self.num_batches = self.data.shape[0] // self.batch_size
        self.reset()

    def reset(self):
        self.batch_no = 0
        np.random.shuffle(self.data) # it's a numpy array

    def get_batch(self):
        start, end = self.batch_no * self.batch_size, (self.batch_no + 1) * self.batch_size
        sentences1 = np.zeros([self.batch_size, 1], dtype=object)
        sentences2 = np.zeros([self.batch_size, 1], dtype=object)
        labels = np.zeros([self.batch_size, 1], dtype=float)
        for batch_idx, data_idx in enumerate(range(start, end)):
            sentences1[batch_idx] = self.data[data_idx][self.sentence1_idx]
            sentences2[batch_idx] = self.data[data_idx][self.sentence1_idx + 4]
            labels[batch_idx] = self.data[data_idx][0]

        return sentences1, sentences2, labels

    def is_finished(self):
        if self.batch_no == self.num_batches - 1:
            self.reset()
            return True
        return False
