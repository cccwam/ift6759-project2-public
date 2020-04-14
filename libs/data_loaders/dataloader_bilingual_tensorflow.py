from abc import ABC
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloaderSubword, create_masks_fm, \
    create_padding_mask_fm
from libs.data_loaders.abstract_dataloader_tensorflow import AbstractTensorFlowTokenizer

logger = tf.get_logger()


class AbstractBilingualTFDataloaderSubword(AbstractBilingualDataloaderSubword, AbstractTensorFlowTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractTensorFlowTokenizer.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[0],
                                   corpus_filename=self._bilingual_corpus_filenames[0])
        self._tokenizer_source: SubwordTextEncoder = res[0]
        self._source_numericalized: List[List[int]] = res[1]

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=self._corpora_filenames[1],
                                   corpus_filename=self._bilingual_corpus_filenames[1])
        self._tokenizer_target: SubwordTextEncoder = res[0]
        self._target_numericalized: List[List[int]] = res[1]

    def decode(self, tokens: List[int]) -> str:
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target)

    @property
    def bos(self) -> int:
        return self._tokenizer_target.encode(self._bos)[0]


class BilingualTranslationTFSubword(AbstractBilingualTFDataloaderSubword):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence and masking.

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualTFDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)

    @staticmethod
    def _my_generator_from_ids(source_numericalized: List[List[int]],
                               target_numericalized: List[List[int]]):
        n_samples = len(source_numericalized)

        for i in range(n_samples):
            yield source_numericalized[i], target_numericalized[i], target_numericalized[i][1:] + [0]

    def _get_train_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_generator_from_ids,
                         source_numericalized=self._source_numericalized,
                         target_numericalized=self._target_numericalized
                         )
        return self._hook_dataset_post_precessing(my_gen=my_gen, batch_size=batch_size)

    def _hook_dataset_post_precessing(self, my_gen, batch_size: int):
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=(tf.int32, tf.int32, tf.int32),
                                            output_shapes=(tf.TensorShape([None]),
                                                           tf.TensorShape([None]),
                                                           tf.TensorShape([None])))
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=([None], [None], [None],))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        def add_mask(source, target_in, target_out):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks_fm(source,
                                                                                target_in)
            return (source, target_in, enc_padding_mask, combined_mask, dec_padding_mask), target_out

        return ds.map(map_func=add_mask)

    def _get_valid_dataset(self, batch_size: int) -> tf.data.Dataset:
        return self._get_train_dataset(batch_size=batch_size)

    def _my_test_generator(self, source_numericalized: List[List[int]]):
        for s in source_numericalized:
            yield s

    def _get_test_dataset(self, batch_size: int) -> tf.data.Dataset:
        lines = self._read_file(corpus_filepath=Path(self._raw_english_test_set_file_path))
        source_numericalized: List[List[int]] = [self._tokenizer_source.encode(s) for s in lines]

        self._test_steps = np.ceil(len(source_numericalized) / self._batch_size)

        my_gen = partial(self._my_test_generator,
                         source_numericalized)
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=tf.int32,
                                            output_shapes=tf.TensorShape([None]))
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=([None]))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        def add_mask(source):
            enc_padding_mask = create_padding_mask_fm(source)
            return source, enc_padding_mask

        return ds.map(map_func=add_mask)
