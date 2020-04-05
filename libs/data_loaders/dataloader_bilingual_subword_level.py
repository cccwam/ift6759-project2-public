import logging
from abc import ABC
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tokenizers import (Encoding)

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader, AbstractBilingualSeq2SeqDataloader, \
    AbstractBilingualTransformersDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = logging.getLogger(__name__)


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

        self._languages: List[str] = self._preprocessed_data_path["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        corpora_filenames: List[List[str]] = self._preprocessed_data_path["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"
        assert len(corpora_filenames) == 2, "You should have only two languages"

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang1_en_tokenized.pickle")
        self._tokenizer_source, self._en_numericalized = res

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[1],
                                   corpus_filename="train_lang2_fr_tokenized.pickle")
        self._tokenizer_target, self._fr_numericalized = res

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
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target)


class BilingualCausalLMDataloaderSubword(AbstractBilingualDataloaderSubword,
                                         AbstractBilingualSeq2SeqDataloader):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence
        and the shifted target sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualSeq2SeqDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = source_numericalized[i].ids
            target = target_numericalized[i].ids
            inputs = (source, target)
            output = target[1:]
            yield (inputs, output)


class BilingualTranslationLMDataloaderSubword(AbstractBilingualDataloaderSubword):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence
        as mask language model, called translation language model in XLM paper.
        https://arxiv.org/abs/1901.07291

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        self._output_types = ((tf.int32, tf.int32, tf.int32),
                              tf.int32)
        self._output_shapes = ((tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
                               tf.TensorShape([None]))
        self._consolidated_seq = self._seq_length_source + self._seq_length_target
        self._padded_shapes = ((self._consolidated_seq, self._consolidated_seq, self._consolidated_seq),
                               self._consolidated_seq)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            # zero is the pad token id
            source = np.zeros([self._seq_length_source], dtype=int)
            source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            # zero is the pad token id
            target = np.zeros([self._seq_length_target], dtype=int)
            target[:len(target_numericalized[i].ids)] = target_numericalized[i].ids

            attention_masks = np.zeros([self._consolidated_seq], dtype=int)
            attention_masks[:len(source_numericalized[i].ids)] = 1

            sent_2_start_idx = len(source_numericalized[i].ids)
            sent_2_end_idx = sent_2_start_idx + len(target_numericalized[i].ids)
            attention_masks[sent_2_start_idx:sent_2_end_idx] = 1

            tokens_type_ids = tf.concat(
                [tf.zeros([self._seq_length_source], dtype=tf.int32),
                 tf.ones([self._seq_length_target], dtype=tf.int32)],
                axis=-1)

            inputs = tf.concat([source, target], axis=-1)
            output = inputs

            yield ((inputs, attention_masks, tokens_type_ids), output)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset):
        # 10% nothing to do, 10% random word, 80% mask
        distrib_mask = tfp.distributions.Multinomial(total_count=1, probs=[0.1, 0.1, 0.8])

        distrib_random = tfp.distributions.Uniform(low=len(self._special_tokens), high=self._vocab_size_source)

        return self._apply_mask_for_mlm(ds=ds,
                                        distrib_mask=distrib_mask,
                                        distrib_random=distrib_random)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target)


# TODO it should not be a subclass of AbstractBilingualDataloader because there is more inputs
class BilingualCustomPretrainingDataloaderSubword(AbstractBilingualDataloaderSubword):
    """
        Dataset for the custom pretraining taks (inspired by XLM translation language model idea):
            - Inputs: Two pairs of sentences, one in English and on in French.
                Tokens are masked like in masked language model
            - Targets:  Predicts the masked tokens and if it's a pair of translated sentence of not (binary)

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        self._output_types = ((tf.int32, tf.int32, tf.int32),
                              (tf.int32, tf.float32,))
        self._output_shapes = ((tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
                               (tf.TensorShape([None]), tf.TensorShape([None])))
        self._consolidated_seq = self._seq_length_source + self._seq_length_target
        self._padded_shapes = ((self._consolidated_seq, self._consolidated_seq, self._consolidated_seq),
                               self._consolidated_seq, 1)

    # TODO add monolingual corpus
    # TODO add also label is translation or not
    # def _my_generator(self,
    #                   source_numericalized: List[Encoding],
    #                   target_numericalized: List[Encoding],
    #                   is_translation: bool):
    #     for i in range(len(source_numericalized)):
    #         # zero is the pad token id
    #         source = np.zeros([self._seq_length_source], dtype=int)
    #         source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids
    #
    #         # zero is the pad token id
    #         target = np.zeros([self._seq_length_target], dtype=int)
    #         target[:len(target_numericalized[i].ids)] = target_numericalized[i].ids
    #
    #         attention_masks = np.zeros([self._consolidated_seq], dtype=int)
    #         attention_masks[:len(source_numericalized[i].ids)] = 1
    #
    #         sent_2_start_idx = len(source_numericalized[i].ids)
    #         sent_2_end_idx = sent_2_start_idx + len(target_numericalized[i].ids)
    #         attention_masks[sent_2_start_idx:sent_2_end_idx] = 1
    #
    #         tokens_type_ids = tf.concat(
    #             [tf.zeros([self._seq_length_source], dtype=tf.int32),
    #              tf.ones([self._seq_length_target], dtype=tf.int32)],
    #             axis=-1)
    #
    #         inputs = tf.concat([source, target], axis=-1)
    #         output = inputs
    #
    #         yield ((inputs, attention_masks, tokens_type_ids, tf.convert_to_tensor(isinstance())), output)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset):
        # 10% nothing to do, 10% random word, 80% mask
        distrib_mask = tfp.distributions.Multinomial(total_count=1, probs=[0.1, 0.1, 0.8])

        distrib_random = tfp.distributions.Uniform(low=len(self._special_tokens), high=self._vocab_size_source)

        return self._apply_mask_for_mlm(ds=ds,
                                        distrib_mask=distrib_mask,
                                        distrib_random=distrib_random)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target)


class BilingualTranslationDataloaderSubword(AbstractBilingualDataloaderSubword):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence
        and masking the input for sentence 2.

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        self._output_types = ((tf.int32, tf.int32, tf.int32),
                              tf.int32)
        self._output_shapes = ((tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
                               tf.TensorShape([None]))
        self._consolidated_seq = self._seq_length_source + self._seq_length_target
        self._padded_shapes = ((self._consolidated_seq, self._consolidated_seq, self._consolidated_seq),
                               self._consolidated_seq)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = np.zeros([self._consolidated_seq], dtype=int)
            source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            target = np.zeros([self._consolidated_seq], dtype=int)
            target_start_idx = self._seq_length_source
            target_end_idx = target_start_idx + len(target_numericalized[i].ids)
            target[target_start_idx:target_end_idx] = target_numericalized[i].ids

            attention_masks = np.zeros([self._consolidated_seq], dtype=int)
            attention_masks[:len(source_numericalized[i].ids)] = 1

            tokens_type_ids = tf.zeros([self._consolidated_seq], dtype=tf.int32)

            yield ((source, attention_masks, tokens_type_ids), target)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens[self._seq_length_source:], tokenizer=self._tokenizer_target)


class BilingualTransformersDataloaderSubword(AbstractBilingualDataloaderSubword,
                                             AbstractBilingualTransformersDataloader):
    """
        Dataset for bilingual corpora at subword level generating only input sentence and target sentence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualTransformersDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = source_numericalized[i].ids
            target = target_numericalized[i].ids
            yield source, target
