import logging
from abc import ABC
from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from tokenizers import (Encoding)

from libs.data_loaders.abstract_dataloader import AbstractMonolingualDataloader, \
    AbstractMonolingualCausalLMDataloader, \
    AbstractMonolingualTransformersLMDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = logging.getLogger(__name__)


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

        self._language: str = self._preprocessed_data_path["language"]
        assert self._language is not None, "Missing language in config"

        monolingual_corpus_filename: str = self._preprocessed_data_path["monolingual_corpus_filename"]
        assert monolingual_corpus_filename is not None, "Missing monolingual_corpus_filename in config"

        corpora_filenames: List[str] = self._preprocessed_data_path["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"

        res = self._load_tokenizer(language=self._language,
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size,
                                   max_seq_length=self._seq_length,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames,
                                   corpus_filename=monolingual_corpus_filename)
        self._tokenizer, self._source_numericalized = res

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
        return self._decode(tokens=tokens, tokenizer=self._tokenizer)


class MonolingualCausalLMDataloaderSubword(AbstractMonolingualDataloaderSubword,
                                           AbstractMonolingualCausalLMDataloader):
    """
        Dataset for monolingual corpora at subword level generating input sentence and the shifted input sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractMonolingualCausalLMDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            inputs = source_numericalized[i].ids
            output = inputs[1:]
            yield (inputs, output)


class MonolingualMaskLMDataloaderSubword(AbstractMonolingualDataloaderSubword,
                                         AbstractMonolingualCausalLMDataloader):
    """
        Dataset for monolingual corpora at subword level generating input sentence and the shifted input sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractMonolingualCausalLMDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            # zero is the pad token id
            inputs = np.zeros([self._seq_length], dtype=int)
            inputs[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            attention_masks = np.zeros([self._seq_length], dtype=int)
            attention_masks[:len(source_numericalized[i].ids)] = 1

            # TODO add management of token_type_ids = 1 for French
            tokens_type_ids = tf.zeros([self._seq_length], dtype=tf.int32)

            yield ((inputs, attention_masks, tokens_type_ids), inputs)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset):
        # 10% nothing to do, 10% random word, 80% mask
        distrib_mask = tfp.distributions.Multinomial(total_count=3, probs=[0.1, 0.1, 0.8])

        distrib_random = tfp.distributions.Uniform(low=len(self._special_tokens), high=self._vocab_size_source)

        return self._apply_mask_for_MLM(ds=ds,
                                        distrib_mask=distrib_mask,
                                        distrib_random=distrib_random)



class MonolingualTransformersLMDataloaderSubword(AbstractMonolingualDataloaderSubword,
                                                 AbstractMonolingualTransformersLMDataloader):
    """
        Dataset for monolingual corpora at subword level generating only input sentence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderSubword.__init__(self, config=config,
                                                      raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractMonolingualTransformersLMDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            yield source_numericalized[i].ids
