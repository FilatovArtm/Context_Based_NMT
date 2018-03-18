import tensor2tensor
from tensorflow.contrib.lookup import MutableHashTable, HashTable, KeyValueTensorInitializer
import tensorflow as tf


class LRUCache:
    def __init__(self, state_size=128, max_size=20):
        self.state_size = state_size
        self.max_size = max_size
        self.default_value_ = 0
        self.attention_cache_ = MutableHashTable(key_dtype=tf.string, value_dtype=tf.float32, default_value=self.default_value_)
        self.state_cache_ = MutableHashTable(key_dtype=tf.string, value_dtype=tf.float32, default_value=self.default_value_)

    def Add(self, key, attention_vector, state_vector):
        """
        Arguments:
            - key: tf.Tensor of shape 1, with dtype=tf.string
            - attention_vector: tf.Tensor(shape=state_size, dtype=tf.float32)
            - state_vector: tf.Tensor(shape=state_size, dtype=tf.float32)
        Returns:
            - void
        """
        tf.cond(tf.equal(self.attention_cache_.size(), 10), true_fn=self.DeleteLRU, false_fn=lambda : 1)
        result = self.attention_cache_.lookup(key)
        tf.cond(tf.not_equal(result[0], -1),
                true_fn=lambda: self.AddExistingKey(key, attention_vector, state_vector),
                false_fn=lambda: self.AddNewKey(key, attention_vector, state_vector)
        )


    def AddExistingKey(self, key, attention_vector, state_vector):
        old_attention_vector = self.attention_cache_.lookup(key)
        old_state_vector = self.state_cache_.lookup(key)
        self.attention_cache_.insert(key, (attention_vector + old_attention_vector) / 2)
        self.state_cache_.insert(key, (state_vector + old_state_vector) / 2)
        
        return True

    def AddNewKey(self, key, attention_vector, state_vector):
        self.attention_cache_.insert(key, attention_vector)
        self.state_cache_.insert(key, state_vector)
        return True


    def DeleteLRU(self):
        """
        Function insert zero vector into least recently used key
        """
        return 0


    def Query(self, attention_vector):
        """
        Arguments:
            - key: tf.Tensor of shape 1, with dtype=tf.string
            - attention_vector: tf.Tensor(shape=state_size, dtype=tf.float32)
            - state_vector: tf.Tensor(shape=state_size, dtype=tf.float32)
        Returns:
            - m_t
        """
        attention_keys, attention_values = self.attention_cache_.export()
        sum = tf.reduce_sum(tf.exp(tf.tensordot(attention_values, attention_vector, 1)))

        state_keys, state_values = self.state_cache_.export()
        weighted_states = tf.tensordot(state_values, tf.tensordot(attention_values, attention_vector, 1) / sum, 1)
        return weighted_states
        