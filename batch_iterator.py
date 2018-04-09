import numpy as np

class BatchIterator():
    def __init__(self, data_iterator, context_size, batch_size, shuffle=True):
        self.context_size = context_size
        self.batch_size = batch_size
        
        # assuming data_iterator either list or generator
        self.raw_data = np.array(list(data_iterator))
        
        # check if we can divide
        assert(len(self.raw_data) % context_size == 0)
        
        self.dataset_size = len(self.raw_data) // context_size
        
        self.indices = np.arange(self.dataset_size)
        
        if shuffle:
            self.indices = np.random.permutation(self.indices)
            
#     def __len__(self):
#         return len(self.raw_data) // self.batch_size + len(self.raw_data) % self.batch_size
        
    def __iter__(self):
        step = 0
        
        for start in range(0, len(self.indices), self.batch_size):
            # select which metaparts are we gonna use for next 
            meta_ix = self.indices[start: start + self.batch_size]

            for context_state in range(self.context_size):
                ix = self.context_size * meta_ix + context_state
                yield self.raw_data[ix]