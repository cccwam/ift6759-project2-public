import logging
import pickle
from abc import ABC
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from libs.data_loaders.abstract_dataloader import AbstractMonolingualDataloader, \
    AbstractMonolingualTransformersLMDataloader, AbstractMonolingualCausalLMDataloader

logger = logging.getLogger(__name__)


class AbstractMonolingualDataloaderWord(AbstractMonolingualDataloader, ABC):
    """
        Dataset for monolingual corpora at word level.

        Max vocab size including special tokens: 91273 for French and 60458 for English.

        Change the vocab size to decrease the # model params

        Special tokens:
        - 0 for MASK
        - 1 for BOS
        - 2 for EOS
        - 3 for UNKNOWN
    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        """
        MonolingualDataloaderWord

        :param config: The configuration dictionary. It must follow configs/user/schema.json
        """
        AbstractMonolingualDataloaderWord.__init__(self, config=config,
                                                   raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._folder: Path = Path(self._preprocessed_data_path["folder"])
        assert self._folder.exists()
        self._monolingual_corpus_filename: str = self._preprocessed_data_path["monolingual_corpus_filename"]
        assert self._monolingual_corpus_filename is not None, "Missing monolingual_corpus_filename in config"
        word_to_token_filename: str = self._preprocessed_data_path["word_to_token_filename"]
        assert word_to_token_filename is not None, "Missing word_to_token_filename in config"

        with open(str(self._folder / self._monolingual_corpus_filename), 'rb') as handle:
            self._source_numericalized = pickle.load(handle)

        with open(str(self._folder / word_to_token_filename), 'rb') as handle:
            self._word_to_token: dict = pickle.load(handle)

        if self._vocab_size is not None:
            # To limit the output vocab size
            self._word_to_token = {k: v for i, (k, v) in enumerate(self._word_to_token.items())
                                   if i < self._vocab_size}

        self._token_to_word: dict = {v: k for k, v in self._word_to_token.items()}
        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(self._source_numericalized)}")

    def _hook_dataset_post_precessing(self, ds):
        if self._vocab_size is not None:
            logger.debug("Vocab size limited to " + str(self._vocab_size))
            # Only to test performance with lower vocab size (and GPU mem)
            ds = ds.map(lambda x, y: (tf.minimum(x, self._vocab_size - 1),
                                      tf.minimum(y, self._vocab_size - 1)))
        return ds

    def get_hparams(self):
        return f"vocab_size_{self._vocab_size}" + \
               f"_seq_length_{self._seq_length}" + \
               f"_corpus_{self._folder / self._monolingual_corpus_filename}"

    def decode(self, tokens):
        mapped_tokens = [self._word_to_token[t] for t in tokens]
        return " ".join(mapped_tokens)


class MonolingualCausalLMDataloaderWord(AbstractMonolingualDataloaderWord,
                                        AbstractMonolingualCausalLMDataloader):
    """
        Dataset for monolingual corpora at word level generating input sentence and the shifted input sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderWord.__init__(self, config=config,
                                                   raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractMonolingualCausalLMDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List):
        bos, eos = 1, 2  # BOS will be 1 and EOS will be 2, leaving MASK to 0 and UNK to 3
        for i in range(len(source_numericalized)):
            inputs = np.array([bos] + source_numericalized[i] + [eos])
            output = inputs[1:]
            yield (inputs, output)


class MonolingualTransformersLMDataloaderWord(AbstractMonolingualDataloaderWord,
                                              AbstractMonolingualTransformersLMDataloader):
    """
        Dataset for monolingual corpora at word level generating only input sentence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractMonolingualDataloaderWord.__init__(self, config=config,
                                                   raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractMonolingualTransformersLMDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List):
        bos, eos = 1, 2  # BOS will be 1 and EOS will be 2, leaving MASK to 0 and UNK to 3
        for i in range(len(source_numericalized)):
            inputs = np.array([bos] + source_numericalized[i] + [eos])
            yield inputs
