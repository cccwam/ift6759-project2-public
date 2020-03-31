import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tokenizers import Encoding

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader, \
    AbstractBilingualTransformersDataloader, AbstractBilingualSeq2SeqDataloader

logger = logging.getLogger(__name__)


class AbstractBilingualDataloaderWord(AbstractBilingualDataloader):
    """
        Dataset for bilingual corpora at word level.

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
        BilingualDataloaderWord

        :param config: The configuration dictionary. It must follow configs/user/schema.json
        """
        AbstractBilingualDataloader.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._folder: Path = Path(self._preprocessed_data_path["folder"])
        assert self._folder.exists()

        with open(str(self._folder / "train_lang1_en_numericalized.pickle"), 'rb') as handle:
            self._en_numericalized = pickle.load(handle)

        with open(str(self._folder / "train_lang2_fr_numericalized.pickle"), 'rb') as handle:
            self._fr_numericalized = pickle.load(handle)

        with open(str(self._folder / "word_to_token_en.pickle"), 'rb') as handle:
            self._word_to_token_en: dict = pickle.load(handle)

        with open(str(self._folder / "word_to_token_fr.pickle"), 'rb') as handle:
            self._word_to_token_fr: dict = pickle.load(handle)

        with open(str(self._folder / "token_to_word_fr.pickle"), 'rb') as handle:
            self._token_to_word_fr: dict = pickle.load(handle)

        with open(str(self._folder / "token_to_word_en.pickle"), 'rb') as handle:
            self._token_to_word_en: dict = pickle.load(handle)

        logger.debug("Vocab size for English limited to " + str(self._vocab_size_source))
        # To limit the output vocab size
        self._token_to_word_en = {k: v for i, (k, v) in enumerate(self._token_to_word_en.items())
                                  if i < self._vocab_size_source}
        logger.debug("Vocab size for French limited to " + str(self._vocab_size_target))
        # To limit the output vocab size
        self._token_to_word_fr = {k: v for i, (k, v) in enumerate(self._token_to_word_fr.items())
                                  if i < self._vocab_size_target}

        logger.debug(f"{str(self.__class__.__name__)} English samples: {len(self._en_numericalized)}")
        logger.debug(f"{str(self.__class__.__name__)} French samples: {len(self._fr_numericalized)})")

    def _hook_dataset_post_precessing(self, ds):
        # Only to test performance with lower vocab size (and GPU mem)
        ds = ds.map(lambda x, y: ((tf.minimum(x[0], self._vocab_size_source - 1),
                                   tf.minimum(x[1], self._vocab_size_target - 1)),
                                  tf.minimum(y, self._vocab_size_target - 1)))
        return ds

    def get_hparams(self):
        return f"vocab_size_{self._vocab_size_source},{self._vocab_size_target}" + \
               f"_seq_length_{self._seq_length_source},{self._seq_length_target}"


class BilingualSeq2SeqDataloaderWord(AbstractBilingualDataloaderWord,
                                     AbstractBilingualSeq2SeqDataloader):
    """
        Dataset for bilingual corpora at word level generating input sentence, target sentence
        and the shifted target sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderWord.__init__(self, config=config,
                                                 raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualSeq2SeqDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List, target_numericalized: List):
        bos, eos = 1, 2  # BOS will be 1 and EOS will be 2, leaving MASK to 0 and UNK to 3
        for i in range(len(source_numericalized)):
            en = np.array([bos] + source_numericalized[i] + [eos])
            fr = np.array([bos] + target_numericalized[i] + [eos])
            inputs = (en, fr)
            output = fr[1:]
            yield (inputs, output)


class BilingualTransformersDataloaderWord(AbstractBilingualDataloaderWord,
                                          AbstractBilingualTransformersDataloader):
    """
        Dataset for bilingual corpora at word level generating only input sentence and target sentence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderWord.__init__(self, config=config,
                                                 raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualTransformersDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            en = source_numericalized[i].ids
            fr = target_numericalized[i].ids
            yield (en, fr)
