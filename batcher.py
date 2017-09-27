import numpy as np

class Batcher:
    def __init__(self, data, batch_size, is_json):
        self.is_json = is_json
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(self.data) // self.batch_size
        self.reset()

    def reset(self):
        self.batch_no = 0
        if self.is_json:
            self.indices = np.random.permutation(len(self.data))
        else: # it's a numpy array
            np.shuffle(self.data)


    def get_batch(self):
        if self.batch_size == 1:
            curr_idx = self.indices[self.batch_no]
            start, end = curr_idx, curr_idx + 1
        else:
            start, end = self.batch_no * args.batch_size, (self.batch_no + 1) * self.batch_size

        if self.is_json:
            sentences1 = self.data[start][4] # these are trees
            sentences2 = self.data[start][8] # these are trees
            labels = self.data[start][0]
        else:
            sentences1 = self.data[start:end, 3]
            sentences2 = self.data[start:end, 4]
            labels = self.data[start:end, 0]

        self.batch_no += 1
        return sentences1, sentences2, labels

    def is_finished(self):
        if self.batch_no == self.num_batches - 1:
            self.reset()
            return True
        return False
