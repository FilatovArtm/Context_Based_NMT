import numpy as np

class LRUCache:
    def __init__(self, max_size=1):
        self.container_ = {}
        self.lru_state_ = []
        self.max_size_ = max_size

    def Query(self, attention_vector):
        result = np.zeros(len(attention_vector))

        sum = 0
        for att_vec, state_vec in self.container_.values():
            sum += np.exp(np.dot(attention_vector, att_vec))

        for att_vec, state_vec in self.container_.values():
            result += state_vec * np.exp(np.dot(att_vec, attention_vector))

        return result / sum

    def Add(self, key, attention_vector, state_vector):
        if len(self.container_) > self.max_size_:
            lru_key = self.lru_state_.pop()
            self.container_.pop(lru_key)

        if key not in self.container_:
            self.container_[key] = [attention_vector, state_vector]
        else:
            old_att, old_state = self.container_[key]
            self.container_[key] = [(old_att + attention_vector) / 2, (old_state + state_vector) / 2]
            self.lru_state_.remove(key)

        self.lru_state_.insert(0, key)
        return attention_vector

