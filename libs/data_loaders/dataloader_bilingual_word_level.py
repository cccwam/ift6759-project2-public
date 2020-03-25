from functools import partial
from pathlib import Path
from abc import ABCMeta, abstractmethod
import pickle
import tensorflow as tf
import numpy as np
import tqdm

from libs.data_loaders import AbstractDataloader
import logging

logger = logging.getLogger(__name__)


class BilingualDataloaderWord(AbstractDataloader):
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

    def __init__(self, config: dict):

        super(BilingualDataloaderWord, self).__init__(config=config)

        with open(self._preprocessed_data_path / "train_lang1_en_numericalized.pickle", 'rb') as handle:
            self._en_numericalized = pickle.load(handle)

        with open(self._preprocessed_data_path / "train_lang2_fr_numericalized.pickle", 'rb') as handle:
            self._fr_numericalized = pickle.load(handle)

        with open(self._preprocessed_data_path / "word_to_token_en.pickle", 'rb') as handle:
            self._word_to_token_en: OrderedDict = pickle.load(handle)

        with open(self._preprocessed_data_path / "word_to_token_fr.pickle", 'rb') as handle:
            self._word_to_token_fr: OrderedDict = pickle.load(handle)

        with open(self._preprocessed_data_path / "token_to_word_fr.pickle", 'rb') as handle:
            self._token_to_word_fr: OrderedDict = pickle.load(handle)

        with open(self._preprocessed_data_path / "token_to_word_en.pickle", 'rb') as handle:
            self._token_to_word_en: OrderedDict = pickle.load(handle)

        if self._vocab_size is not None:
            logger.debug("Vocab size limited to " + str(self._vocab_size))
            # To limit the output vocab size
            self._token_to_word_fr = {k: v for i, (k, v) in enumerate(self._token_to_word_fr.items())
                                      if i < self._vocab_size}

        logger.debug(f"{str(self.__class__.__name__)} English samples: {len(self._en_numericalized)}")
        logger.debug(f"{str(self.__class__.__name__)} French samples: {len(self._fr_numericalized)})")

    @classmethod
    def _my_generator(cls, en_numericalized, fr_numericalized):
        bos, eos = 1, 2  # BOS will be 1 and EOS will be 2, leaving MASK to 0 and UNK to 3
        for i in range(len(en_numericalized)):
            en = np.array([bos] + en_numericalized[i] + [eos])
            fr = np.array([bos] + fr_numericalized[i] + [eos])
            inputs = (en, fr)
            output = fr[1:]
            yield (inputs, output)

    def build(self,
              batch_size):

        my_gen = partial(BilingualDataloaderWord._my_generator,
                         self._en_numericalized,
                         self._fr_numericalized)

        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=((tf.int32, tf.int32), tf.int32),
                                            output_shapes=((tf.TensorShape([None]), tf.TensorShape([None])),
                                                           tf.TensorShape([None])))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if self._vocab_size is not None:
            # Only to test performance with lower vocab size (and GPU mem)
            ds = ds.map(lambda x, y: ((tf.minimum(x[0], self._vocab_size - 1),
                                       tf.minimum([1], self._vocab_size - 1)),
                                      tf.minimum(y, self._vocab_size - 1)))

        ds = ds.padded_batch(batch_size=batch_size, padded_shapes=(([self._seq_length], [self._seq_length]),
                                                                   self._seq_length))

        self._build_all_dataset(ds, batch_size)
