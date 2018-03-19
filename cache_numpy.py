import numpy as np

class LRUCache:
    def __init__(self, max_size=1):
        self.container_ = {}
        self.max_size_ = max_size

    def Query(self, attention_vector):
        result = np.zeros(len(attention_vector))
        for vector in self.container_.values():
            result += vector

        return result

    def Add(self, key, attention_vector):
        if len(self.container_) > self.max_size_:
            self.container_.pop()
        self.container_[key] = attention_vector
        return attention_vector
