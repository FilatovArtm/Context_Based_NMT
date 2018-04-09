import numpy as np
import tensorflow as tf

class LRUCache:
    def __init__(self, hidden_size, max_size=10, batch_size=1):
        '''
        attention_tensor - reference for tensor of size (batch_size x max_size x hidden_size)
        initialize with zeros
        '''
        self.attention_tensor_ = tf.zeros((batch_size, max_size, hidden_size))
        self.state_tensor_ = tf.zeros((batch_size, max_size, hidden_size))
        self.max_size_ = max_size
        self.batch_size_ = batch_size
        self.mapping_ = []
        for _ in range(batch_size):
            self.mapping_.append(dict())

        self.lru_state_ = [[] for i in range(batch_size)]

    def Query(self, matching_vectors):
        '''
        matching_vector: tensor of size (batch_size x hidden_size) 
        '''

        weights = tf.nn.softmax(
            tf.einsum("ijk, ik->ij", cache.state_tensor_, matching_vectors), dim=1)
        return tf.einsum("ijk, ij->ik", cache.attention_tensor_, weights)

    def Add(self, tokens, state_vectors, attention_vectors):
        indeces, alphas = tf.py_func(self._AddPy, [tokens], (tf.int64, tf.float32))

        self.state_tensor_[tf.range(self.batch_size_)[:, None], indeces] = \
            state_vectors * alphas[:, :, None] + \
            self.state_tensor_[tf.range(self.batch_size_)[
                :, None], indeces] * (1 - alphas[:, :, None])

        self.attention_tensor_[tf.range(self.batch_size_)[:, None], indeces] = \
            attention_vectors * alphas[:, :, None] + \
            self.attention_tensor_[tf.range(self.batch_size_)[
                :, None], indeces] * (1 - alphas[:, :, None])
        return tf.float32(0)

    def _AddEntry(self, num_batch, token):
        '''
        Returns tensor of size (2, ) with index and alpha params
        '''
        lru_index = len(self.mapping_[num_batch])
        if len(self.mapping_[num_batch]) == self.max_size_:
            lru_key = self.lru_state_[batch_size].pop()
            lru_index = self.mapping_[num_batch].pop(lru_key)

        if token not in self.mapping_[num_batch]:
            self.mapping_[num_batch][token] = lru_index
            index = lru_index
            alpha = 1.
        else:
            index = self.mapping_[num_batch][token]
            alpha = 0.5
            self.lru_state_[num_batch].remove(token)

        self.lru_state_[num_batch].insert(0, token)
        return index, alpha

    def _AddPy(self, keys_batch):
        '''
        keys_batch: tensor with token of size (batch_size x sentence_size)

        return: tensor of size (batch_size x sentence_size)
        '''
        indeces = np.zeros((keys_batch.shape))
        alphas = np.zeros((keys_batch.shape))

        for i in range(self.batch_size_):
            for token_ind in range(len(keys_batch[i])):

                indeces[i, token_ind], alphas[i, token_ind] = self._AddEntry(
                    i, keys_batch[i][token_ind])

        return indeces.astype(np.int64), alphas
