# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models.transformer import Transformer, transformer_prepare_decoder, transformer_decoder

import tensorflow as tf

from tensorflow.python.util import nest
import numpy as np


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(features[key], 1.0)
  return None


import numpy as np

class LRUCache:
    def __init__(self, max_size=1):
        self.container_ = {}
        self.lru_state_ = []
        self.max_size_ = max_size

    def Flush(self):
      self.container_ = {}
      self.lru_state_ = []
      
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
            self.container_[key] = state_vector / np.linalg.norm(state_vector)
        else:
            old_state = self.container_[key]
            self.container_[key] = (old_state + state_vector) / 2
            self.container_[key] /= np.linalg.norm(self.container_[key])
            self.lru_state_.remove(key)

        self.lru_state_.insert(0, key)
        return np.int64(0)

    def AddMultipleEntries(self, keys, state_vectors):
        keys = keys.ravel()
        reshaped_states = state_vectors.reshape(len(keys), -1)
        for key, state_vector in zip(keys, reshaped_states):
            self.Add(key, state_vector)

        return np.int64(0)



@registry.register_model
class TransformerCache(Transformer):
  """Attention net.  See file docstring."""
  def __init__(self, *args, **kwargs):
    super(TransformerCache, self).__init__(*args, **kwargs)
    self.attention_weights = dict()  # For vizualizing attention heads.
    self.sentence_cache = LRUCache(20)

    with tf.variable_scope("sentence_level_cache"):
      self.m_weight = tf.get_variable(
        'm_weight', shape=[self.hparams.hidden_size, self.hparams.hidden_size]
      )
      self.s_weight = tf.get_variable(
        's_weight', shape=[self.hparams.hidden_size, self.hparams.hidden_size]
      )
      self.cache_flag = tf.Variable(0, trainable=False, name='flag', dtype=tf.int64)

  def calculate_mixing_weight(self, s, m):
    return tf.einsum(
        'jl,ikl->ikj', self.m_weight, m
      ) + tf.einsum(
        'jl,ikl->ikj', self.s_weight, s
      )



  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             nonpadding=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights)

    if (common_layers.is_on_tpu() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      m = tf.py_func(self.sentence_cache.QueryMultipleEntries, [decoder_output], tf.float32)
      m.set_shape(decoder_output.get_shape())
      lambd = self.calculate_mixing_weight(decoder_output, m)
      return tf.expand_dims(lambd * decoder_output + (1.0 - lambd) * m, axis=2)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)

    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets")
    )

    self.cache_flag = tf.py_func(
      self.sentence_cache.AddMultipleEntries,
      [features["targets_raw"], decoder_output],
      tf.int64,
    )

    tf.cast(self.cache_flag, tf.float32)


    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}
    self.cache_flag.set_shape((1,))
    return decoder_output + 0 * self.cache_flag


  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   sentence_cache=None):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.target_modality

    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
        decode_length = 1
      else:
        decode_length = common_layers.shape_list(inputs)[1] + decode_length

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.input_modality["inputs"]
      with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode, inputs, features["target_space_id"], hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      # In this case, features["inputs"] contains partial targets.
      # We force the outputs to begin with these sequences. 
      encoder_output = None
      encoder_decoder_attention_bias = None
      partial_targets = tf.squeeze(tf.to_int64(features["inputs"]), [2, 3])
      partial_targets_length = common_layers.shape_list(partial_targets)[1]
      decode_length += partial_targets_length
      batch_size = tf.shape(partial_targets)[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode, targets, cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias, hparams, cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]
        def forced_logits():
          return tf.one_hot(tf.tile(partial_targets[:, i], [beam_size]),
                            vocab_size, 0.0, -1e9)
        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache, body_outputs

    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        sentence_cache=self.sentence_cache,
        cache_flag=self.cache_flag)
    if partial_targets is not None:
      ret["outputs"] = ret["outputs"][:, partial_targets_length:]
    return ret


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                sentence_cache=None,
                cache_flag=None):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for slonger translations.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

  cache = {
      "layer_%d" % layer: {
          "k": tf.zeros([batch_size, 0, key_channels]),
          "v": tf.zeros([batch_size, 0, value_channels]),
      }
      for layer in range(num_layers)
  }

  if encoder_output is not None:
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    decoded_ids, scores = beam_search.beam_search(
        lambda x : symbols_to_logits_fn(x)[:-1],
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
  else:  # Greedy

    def inner_loop(cache_flag, i, finished, next_id, decoded_ids, cache):
      """One step of greedy decoding."""
      logits, cache, out = symbols_to_logits_fn(next_id, i, cache)
      temperature = (0.0 if hparams.sampling_method == "argmax" else
                     hparams.sampling_temp)
      next_id = common_layers.sample_with_temperature(logits, temperature)
      finished |= tf.equal(next_id, eos_id)
      next_id = tf.expand_dims(next_id, axis=1)
      
      cache_flag = tf.py_func(sentence_cache.AddMultipleEntries, [next_id, out], tf.int64)
      cache_flag.set_shape(tf.TensorShape([]))
      
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return cache_flag, i + 1, finished, next_id, decoded_ids, cache

    def is_not_finished(cache_flag, i, finished, *_):
      return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    finished = tf.fill([batch_size], False)
    next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
    cache_flag, _, _, _, decoded_ids, _ = tf.while_loop(
        is_not_finished,
        inner_loop,
        [cache_flag, tf.constant(0), finished, next_id, decoded_ids, cache],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
        ])
    scores = None

  return {"outputs": decoded_ids + cache_flag, "scores": scores}