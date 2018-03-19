import numpy as np

class LRUCache:
    def __init__(self, max_size=1):
        self.container_ = {}
        self.lru_state_ = []
        self.max_size_ = max_size

    def Query(self, matching_vector):
        result = np.zeros(len(matching_vector))

        dot_products = []
        for state_vec in self.container_.values():
            dot_products.append(np.dot(matching_vector, state_vec))

        dot_products = np.array(dot_products)
        sum = 1
        if len(dot_products) > 0:
            dot_products -= np.max(dot_products)
            sum = np.sum(np.exp(dot_products))

        for state_vec, product in zip(self.container_.values(), dot_products):
            result += state_vec * np.exp(product)

        return result / sum


    def QueryMultipleEntries(self, matching_vectors):
        num_entries = matching_vectors.shape[1]
        results = np.zeros_like(matching_vectors, dtype=np.float32)

        for i in range(num_entries):
            results[0, i] = self.Query(matching_vectors[0, i])

        return results

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
        return np.float32(0.)

    def AddMultipleEntries(self, keys, state_vectors):
        keys = keys.ravel()
        reshaped_states = state_vectors.reshape(len(keys), -1)
        for key, state_vector in zip(keys, reshaped_states):
            self.Add(key, state_vector)

        return np.float32(0.)

