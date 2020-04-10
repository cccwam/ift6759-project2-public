from abc import ABC
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tokenizers import (Encoding)
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = tf.get_logger()


class AbstractBilingualDataloaderSubword(AbstractBilingualDataloader, AbstractHuggingFacesTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloader.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractHuggingFacesTokenizer.__init__(self, config=config,
                                               raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._languages: List[str] = self._preprocessed_data["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        corpora_filenames: List[List[str]] = self._preprocessed_data["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"
        assert len(corpora_filenames) == 2, "You should have only two languages"

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang1_en_tokenized.pickle",
                                   is_training=True)
        self._tokenizer_source_training: BaseTokenizer = res[0]
        self._source_with_dropout: List[str] = res[1]

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang1_en_tokenized.pickle",
                                   is_training=False)
        self._tokenizer_source_inference: BaseTokenizer = res[0]
        self._source_without_dropout: List[Encoding] = res[1]

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[1],
                                   corpus_filename="train_lang2_fr_tokenized.pickle",
                                   is_training=True)
        self._tokenizer_target_training: BaseTokenizer = res[0]
        self._target_with_dropout: List[str] = res[1]

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[1],
                                   corpus_filename="train_lang2_fr_tokenized.pickle",
                                   is_training=False)
        self._tokenizer_target_inference: BaseTokenizer = res[0]
        self._target_without_dropout: List[Encoding] = res[1]

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._languages[0],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_source,
                                                   dropout=self._dropout) + "-" + \
               self._get_tokenizer_filename_prefix(language=self._languages[1],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_target,
                                                   dropout=self._dropout)

    def decode(self, tokens: List[int]):
        tokens_until_eos = []
        for token in tokens:
            if token == 3:  # EOS
                break
            tokens_until_eos += [token]

        return self._decode(tokens=tokens_until_eos, tokenizer=self._tokenizer_target_inference)


class BilingualTranslationSubword(AbstractBilingualDataloaderSubword):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence and masking.

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        self._output_types = ((tf.int32, tf.int32, tf.float32, tf.float32, tf.float32),
                              tf.int32)
        self._output_shapes = ((tf.TensorShape([None]),
                                tf.TensorShape([None]),
                                tf.TensorShape([1, 1, self._seq_length_source]),
                                tf.TensorShape([1, self._seq_length_target, self._seq_length_target]),
                                tf.TensorShape([1, 1, self._seq_length_target])),
                               tf.TensorShape([None]))
        self._padded_shapes = ((self._seq_length_source, self._seq_length_source,
                                (1, 1, self._seq_length_source),
                                (1, self._seq_length_target, self._seq_length_target),
                                (1, 1, self._seq_length_target)),
                               self._seq_length_target)

        self._output_test_types = (tf.int32, tf.float32)
        self._output_test_shapes = (tf.TensorShape([None]),
                                    tf.TensorShape([1, 1, self._seq_length_source]))
        self._padded_test_shapes = (self._seq_length_source, (1, 1, self._seq_length_source))

    def _my_generator_from_strings(self, source: List[str], target: List[str],
                                   tokenizer_source: BaseTokenizer, tokenizer_target: BaseTokenizer):
        source_numericalized: List[Encoding] = tokenizer_source.encode_batch(source)
        target_numericalized: List[Encoding] = tokenizer_target.encode_batch(target)
        return self._my_generator_from_encodings(source_numericalized=source_numericalized,
                                                 target_numericalized=target_numericalized)

    def _my_generator_from_encodings(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        batch_size = len(source_numericalized)

        # TF version is slower than NP
        # source = tf.Variable(tf.zeros([self._seq_length_source], dtype=tf.int32))
        # target_in = tf.Variable(tf.zeros([self._seq_length_target], dtype=tf.int32))
        # target_out = tf.Variable(tf.zeros([self._seq_length_target], dtype=tf.int32))
        #
        # for i in range(batch_size):
        #     source[:len(source_numericalized[i].ids)].assign(source_numericalized[i].ids)
        #
        #     target_in[0:len(target_numericalized[i].ids)].assign(target_numericalized[i].ids)
        #
        #     target_out[0:len(target_numericalized[i].ids) - 1].assign(target_numericalized[i].ids[1:])
        #
        #     enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(source, target_in)
        #
        #     yield ((source.value(), target_in.value(), enc_padding_mask, combined_mask, dec_padding_mask),
        #            target_out.value())

        source = np.zeros([self._seq_length_source], dtype=int)
        target_in = np.zeros([self._seq_length_target], dtype=int)
        target_out = np.zeros([self._seq_length_target], dtype=int)

        for i in range(batch_size):
            source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            target_in[0:len(target_numericalized[i].ids)] = target_numericalized[i].ids

            target_out[0:len(target_numericalized[i].ids) - 1] = target_numericalized[i].ids[1:]

            enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(source, target_in)

            yield ((source, target_in, enc_padding_mask, combined_mask, dec_padding_mask), target_out)

    def _my_test_generator(self, test_input_file: Path):

        with test_input_file.open() as file:
            lines = [self._bos + line + self._eos for line in file.readlines()]
        source_numericalized: List[Encoding] = self._tokenizer_source_inference.encode_batch(sequences=lines)

        for i in range(len(source_numericalized)):
            source = np.zeros([self._seq_length_source], dtype=int)
            source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            enc_padding_mask = self._create_padding_mask(source)
            yield (source, enc_padding_mask)

    def _get_train_generator(self):
        return partial(self._my_generator_from_strings,
                       self._source_with_dropout,
                       self._target_with_dropout,
                       self._tokenizer_source_training,
                       self._tokenizer_target_training)

    def _get_valid_generator(self):
        return partial(self._my_generator_from_encodings,
                       self._source_without_dropout,
                       self._target_without_dropout)

    def _get_test_generator(self):
        return partial(self._my_test_generator,
                       Path(self._raw_english_test_set_file_path))

    def build(self,
              batch_size):
        """
            This fundction is overkill but it prevents impacting the logic from libs.models.transformerv2
        Args:
            batch_size:

        Returns:

        """
        if self._raw_english_test_set_file_path is None:
            AbstractBilingualDataloaderSubword.build(self, batch_size=batch_size)
        else:
            self.build_test_dataset(batch_size=batch_size)
