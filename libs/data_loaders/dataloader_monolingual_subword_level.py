from abc import ABC
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
from tokenizers import (Encoding)
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractMonolingualDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = tf.get_logger()


class AbstractMonolingualDataloaderSubword(AbstractMonolingualDataloader, AbstractHuggingFacesTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for monolingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloader.__init__(self, config=config,
                                               raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractHuggingFacesTokenizer.__init__(self, config=config,
                                               raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._language: str = self._preprocessed_data["language"]
        assert self._language is not None, "Missing language in config"

        monolingual_corpus_filename: str = self._preprocessed_data["monolingual_corpus_filename"]
        assert monolingual_corpus_filename is not None, "Missing monolingual_corpus_filename in config"

        corpora_filenames: List[str] = self._preprocessed_data["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"

        res = self._load_tokenizer(language=self._language,
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size,
                                   max_seq_length=self._seq_length,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames,
                                   corpus_filename=monolingual_corpus_filename,
                                   is_training=True)
        self._tokenizer_training, self._source_with_dropout = res

        res = self._load_tokenizer(language=self._language,
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size,
                                   max_seq_length=self._seq_length,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames,
                                   corpus_filename=monolingual_corpus_filename,
                                   is_training=False)
        self._tokenizer_inference, self._source_without_dropout = res

    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float):
        return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._language,
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size,
                                                   dropout=self._dropout)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_inference)


class MonolingualMaskLMDataloaderSubword(AbstractMonolingualDataloaderSubword):
    """
        Dataset for monolingual corpora at subword level
            - Inputs: One sentence in one language with mask for pad tokens and with some masked or replacement as per
            marsked language model task
            - Targets:  Predicts the masked tokens
    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._output_types = ((tf.int32, tf.float32),
                              tf.int32)
        self._output_shapes = ((tf.TensorShape([None]), tf.TensorShape([1, 1, self._seq_length])),
                               tf.TensorShape([None]))
        self._padded_shapes = ((self._seq_length, (1, 1, self._seq_length)),
                               self._seq_length)

    def _my_generator_from_strings(self, source: List[str],
                                   tokenizer_source: BaseTokenizer):
        source_numericalized: List[Encoding] = tokenizer_source.encode_batch(source)
        return self._my_generator_from_encodings(source_numericalized=source_numericalized)

    def _my_generator_from_encodings(self, source_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            # zero is the pad token id
            inputs = np.zeros([self._seq_length], dtype=int)
            inputs[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            enc_padding_mask = self._create_padding_mask(inputs)

            yield ((inputs, enc_padding_mask), inputs)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset, is_training: bool):
        # For masked language model task, all datasets are masked
        return self._apply_mask_for_mlm(ds=ds,
                                        vocab_size=self._vocab_size)

    def _get_train_generator(self):
        return partial(self._my_generator_from_strings,
                       self._source_with_dropout,
                       self._tokenizer_training)

    def _get_valid_generator(self):
        return partial(self._my_generator_from_encodings,
                       self._source_without_dropout)

    def _get_test_generator(self):
        return partial(self._my_generator_from_encodings,
                       self._source_without_dropout)
