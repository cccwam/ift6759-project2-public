from abc import ABC
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tokenizers import (Encoding)
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractMonolingualDataloader, create_padding_mask_fm
from libs.data_loaders.abstract_dataloader_huggingface import AbstractHuggingFaceTokenizer

logger = tf.get_logger()


class AbstractMonolinguaHFlDataloaderSubword(AbstractMonolingualDataloader, AbstractHuggingFaceTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for monolingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloader.__init__(self, config=config,
                                               raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractHuggingFaceTokenizer.__init__(self, config=config,
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
        self._tokenizer_training: BaseTokenizer = res[0]
        self._source_with_dropout: List[List[str]] = res[1]

        res = self._load_tokenizer(language=self._language,
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size,
                                   max_seq_length=self._seq_length,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames,
                                   corpus_filename=monolingual_corpus_filename,
                                   is_training=False)
        self._tokenizer_inference: BaseTokenizer = res[0]
        self._source_without_dropout: List[Encoding] = res[1]

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._language,
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size,
                                                   dropout=self._dropout)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_inference)

    @property
    def bos(self) -> int:
        return self._tokenizer_inference.encode(self._bos).ids[0]


class MonolingualMaskedLanguageModelHF(AbstractMonolinguaHFlDataloaderSubword):
    """
        Dataset for monolingual corpora at subword level
            - Inputs: One sentence in one language with mask for pad tokens and with some masked or replacement as per
            marsked language model task
            - Targets:  Predicts the masked tokens
    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolinguaHFlDataloaderSubword.__init__(self, config=config,
                                                        raw_english_test_set_file_path=raw_english_test_set_file_path)

    def _my_generator_from_strings(self, source: List[str],
                                   tokenizer_source: BaseTokenizer):
        source_numericalized: List[Encoding] = tokenizer_source.encode_batch(source)

        return self._my_generator_from_encodings(source_numericalized=source_numericalized)

    def _get_train_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_generator_from_strings,
                         self._source_with_dropout,
                         self._tokenizer_training)
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=tf.int32,
                                            output_shapes=tf.TensorShape([None]))
        return self._hook_dataset_post_precessing(ds=ds, batch_size=batch_size)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset, batch_size: int):
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=[None])
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        def add_padding_mask(source):
            enc_padding_mask = create_padding_mask_fm(source)
            return (source, enc_padding_mask), source

        ds = ds.map(map_func=add_padding_mask)  # Add pad mask

        # For masked language model task, all datasets are masked
        return self._apply_mask_for_mlm(ds=ds, vocab_size=self._vocab_size)

    def _apply_mask_for_mlm(self,
                            ds: tf.data.Dataset,
                            vocab_size: int):
        """
            Apply mask for masked language model

        Args:
            ds: dataset
            vocab_size: vocab size

        Returns:

        """

        # Do action only for 15% of tokens (and mask output for others)
        prob_mask_idx = 0.15
        # 10% nothing to do, 10% random word, 80% mask
        # prob_nothing, prob_random_replacement, prob_replace_by_mask \
        prob_mask_actions = np.array([0.1, 0.1, 0.8])
        prob_mask_actions = prob_mask_actions * prob_mask_idx
        prob_mask_actions = np.append(prob_mask_actions, [1 - sum(prob_mask_actions)]).tolist()

        distrib_mask = tfp.distributions.Multinomial(total_count=1,
                                                     probs=prob_mask_actions)

        @tf.function
        def apply_mask(x, output):
            inputs, enc_padding_mask = x

            input_shape = tf.shape(inputs)  # Batch size * Seq Length
            output_shape = tf.shape(output)  # Batch size * Seq Length

            masks = distrib_mask.sample(input_shape,
                                        seed=self._seed)  # Batch size *Seq Length * Probability for each class (4)
            masks = tf.cast(masks, dtype=tf.int32)

            random_tokens = tf.random.uniform(input_shape, minval=len(self._special_tokens), maxval=vocab_size,
                                              dtype=inputs.dtype, seed=self._seed, name=None)

            # Replace with mask
            # One is the mask token id for HuggingFace tokenizers
            inputs_masked = tf.where(tf.math.equal(masks[:, :, 2], 1), inputs, tf.ones(input_shape, dtype=inputs.dtype))

            # Replace with random token
            inputs_masked = tf.where(tf.math.equal(masks[:, :, 1], 1), inputs_masked, random_tokens)

            output_masked = tf.where(tf.math.equal(masks[:, :, 3], 1),
                                     tf.zeros(output_shape, dtype=output.dtype),
                                     output)

            return (inputs_masked, enc_padding_mask), output_masked

        return ds.map(map_func=apply_mask)

    @staticmethod
    def _my_generator_from_encodings(source_numericalized: List[Encoding]):
        n_samples = len(source_numericalized)

        for i in range(n_samples):
            yield source_numericalized[i].ids

    def _get_valid_dataset(self, batch_size: int) -> tf.data.Dataset:
        my_gen = partial(self._my_generator_from_encodings,
                         self._source_without_dropout)
        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=tf.int32,
                                            output_shapes=tf.TensorShape([None]))
        return self._hook_dataset_post_precessing(ds=ds, batch_size=batch_size)

    def _get_test_dataset(self, batch_size: int) -> tf.data.Dataset:
        raise NotImplementedError("No test set for pretraining task")
