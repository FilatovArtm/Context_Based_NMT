import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate


_INPUT_FILES_BPE = [
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/en_test.src',
        ('en_test.src',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/en_dev.src',
        ('en_dev.src',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/en_train.src',
        ('en_train.src',)
    ]
]

_OUTPUT_FILES_BPE = [
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/ru_test.dst',
        ('ru_test.dst',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/ru_dev.dst',
        ('ru_dev.dst',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev_bpe/ru_train.dst',
        ('ru_train.dst',)
    ]
]


_INPUT_FILES = [
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/en_test.src',
        ('en_test.src',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/en_dev.src',
        ('en_dev.src',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/en_train.src',
        ('en_train.src',)
    ]
]

_OUTPUT_FILES = [
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/ru_test.dst',
        ('ru_test.dst',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/ru_dev.dst',
        ('ru_dev.dst',)
    ],
    [
        '/workspace/MT/shad_nlp18_contextNMT/data_4prev/ru_train.dst',
        ('ru_train.dst',)
    ]
]

_VOCAB_IN = 'vocab_enru.src'
_VOCAB_OUT = 'vocab_enru.dst'


@registry.register_problem
class ShadENRUOpusProblem(translate.TranslateProblem):

    @property
    def approx_vocab_size(self):
        return 2**15  # 32k

    @property
    def source_vocab_name(self):
        return "vocab_enru.src"

    @property
    def target_vocab_name(self):
        return "vocab_enru.dst"

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        
        train = dataset_split == problem.DatasetSplit.TRAIN
        
        source_datasets = _INPUT_FILES
        target_datasets = _OUTPUT_FILES
        
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.source_vocab_name,
            self.approx_vocab_size,
            source_datasets,
            file_byte_budget=1e8)
        
        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.target_vocab_name,
            self.approx_vocab_size,
            target_datasets,
            file_byte_budget=1e8)
        
        tag = "train" if train else "dev"
        
        filename_src = "en_{}.src".format(tag)
        filename_dst = "ru_{}.dst".format(tag)
        
        data_path = './shad_nlp18_contextNMT/data_4prev/'
        
        return text_problems.text2text_generate_encoded(
            text_problems.text2text_txt_iterator(
                data_path + filename_src,
                data_path + filename_dst
            ),
            source_vocab, target_vocab
        )

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }