from abc import ABC
from functools import partial
from pathlib import Path
from typing import List

import tensorflow as tf
from tokenizers import (Encoding)
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloaderSubword
from libs.data_loaders.abstract_dataloader_huggingface import AbstractHuggingFaceTokenizer

logger = tf.get_logger()


class AbstractBilingualHFDataloaderSubword(AbstractBilingualDataloaderSubword, AbstractHuggingFaceTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractHuggingFaceTokenizer.__init__(self, config=config,
                                              raw_english_test_set_file_path=raw_english_test_set_file_path)

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[0],
                                   corpus_filename=self._bilingual_corpus_filenames[0],
                                   is_training=True)
        self._tokenizer_source_training: BaseTokenizer = res[0]
        self._source_with_dropout: List[str] = res[1]

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[0],
                                   corpus_filename=self._bilingual_corpus_filenames[0],
                                   is_training=False)
        self._tokenizer_source_inference: BaseTokenizer = res[0]
        self._source_without_dropout: List[Encoding] = res[1]

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[1],
                                   corpus_filename=self._bilingual_corpus_filenames[1],
                                   is_training=True)
        self._tokenizer_target_training: BaseTokenizer = res[0]
        self._target_with_dropout: List[str] = res[1]

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[1],
                                   corpus_filename=self._bilingual_corpus_filenames[1],
                                   is_training=False)
        self._tokenizer_target_inference: BaseTokenizer = res[0]
        self._target_without_dropout: List[Encoding] = res[1]

    def decode(self, tokens: List[int]) -> str:
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target_inference)

    @property
    def bos(self) -> int:
        return self._tokenizer_target_training.encode(self._bos)[0]


class BilingualTranslationHFSubword(AbstractBilingualHFDataloaderSubword):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence and masking.

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualHFDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)

    def _my_generator_from_strings(self, source: List[str], target: List[str],
                                   tokenizer_source: BaseTokenizer, tokenizer_target: BaseTokenizer):
        source_numericalized: List[Encoding] = tokenizer_source.encode_batch(source)
        target_numericalized: List[Encoding] = tokenizer_target.encode_batch(target)
        return self._my_generator_from_encodings(source_numericalized=source_numericalized,
                                                 target_numericalized=target_numericalized)

    def _get_train_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_generator_from_strings,
                         self._source_with_dropout,
                         self._target_with_dropout,
                         self._tokenizer_source_training,
                         self._tokenizer_target_training)
        return self._hook_dataset_post_precessing(my_gen=my_gen, batch_size=batch_size)

    def _hook_dataset_post_precessing(self, my_gen, batch_size: int):
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=(tf.int32, tf.int32, tf.int32),
                                            output_shapes=(tf.TensorShape([None]),
                                                           tf.TensorShape([None]),
                                                           tf.TensorShape([None])))
        # noinspection DuplicatedCode
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=([None], [None], [None],))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        def add_mask(source, target_in, target_out):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(source,
                                                                                  target_in)
            return (source, target_in, enc_padding_mask, combined_mask, dec_padding_mask), target_out

        return ds.map(map_func=add_mask)

    def _my_generator_from_encodings(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        n_samples = len(source_numericalized)

        for i in range(n_samples):
            yield source_numericalized[i].ids, target_numericalized[i].ids, target_numericalized[i].ids[1:] + [0]

    def _get_valid_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_generator_from_encodings,
                         self._source_without_dropout,
                         self._target_without_dropout)
        return self._hook_dataset_post_precessing(my_gen=my_gen, batch_size=batch_size)

    def _my_test_generator(self, test_input_file: Path):

        lines = self._read_file(corpus_filepath=test_input_file)
        source_numericalized: List[Encoding] = self._tokenizer_source_inference.encode_batch(sequences=lines)

        for i in range(len(source_numericalized)):
            yield source_numericalized[i].ids

    def _get_test_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_test_generator,
                         Path(self._raw_english_test_set_file_path))
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=tf.int32,
                                            output_shapes=tf.TensorShape([None]))
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=([None]))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        def add_mask(source):
            enc_padding_mask = self._create_padding_mask(source)
            return source, enc_padding_mask

        return ds.map(map_func=add_mask)
