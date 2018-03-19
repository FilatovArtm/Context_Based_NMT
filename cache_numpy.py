import numpy as np

class LRUCache:
    def __init__(self, max_size=1):
        self.container_ = {}
        self.lru_state_ = []
        self.max_size_ = max_size

    def Query(self, matching_vector):
        result = np.zeros(len(matching_vector))

        sum = 0
        for state_vec in self.container_.values():
            sum += np.exp(np.dot(matching_vector, state_vec))

        for state_vec in self.container_.values():
            result += state_vec * np.exp(np.dot(state_vec, matching_vector))

        return result / sum

    def Add(self, key, state_vector):
        if len(self.container_) == self.max_size_:
            lru_key = self.lru_state_.pop()
            self.container_.pop(lru_key)

        if key not in self.container_:
            self.container_[key] = state_vector
        else:
            old_state = self.container_[key]
            self.container_[key] = (old_state + state_vector) / 2
            self.lru_state_.remove(key)

        self.lru_state_.insert(0, key)
        return state_vector

    def AddMultipleEntries(self, keys, state_vectors):
        keys = keys.ravel()
        reshaped_states = state_vectors.reshape(len(keys), -1)
        for key, state_vector in zip(keys, reshaped_states):
            self.Add(key, state_vector)

        return state_vectors

