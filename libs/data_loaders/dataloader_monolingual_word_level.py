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


class MonolingualDataloaderWord(AbstractDataloader):
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

    def __init__(self, config: dict):

        super(MonolingualDataloaderWord, self).__init__(config=config)

        monolingual_corpus_filename: str = self._dl_hparams["monolingual_corpus_filename"]
        assert monolingual_corpus_filename is not None, "Missing monolingual_corpus_filename in config"
        word_to_token_filename: str = self._dl_hparams["word_to_token_filename"]
        assert word_to_token_filename is not None, "Missing word_to_token_filename in config"

        with open(self._preprocessed_data_path / monolingual_corpus_filename, 'rb') as handle:
            self._source_numericalized = pickle.load(handle)

        with open(self._preprocessed_data_path / word_to_token_filename, 'rb') as handle:
            self._word_to_token: OrderedDict = pickle.load(handle)

        if self._vocab_size is not None:
            # To limit the output vocab size
            self._word_to_token = {k: v for i, (k, v) in enumerate(self._word_to_token.items())
                                   if i < self._vocab_size}

        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(self._source_numericalized)}")

    @classmethod
    def _my_causal_lm_generator(cls, source_numericalized):
        bos, eos = 1, 2  # BOS will be 1 and EOS will be 2, leaving MASK to 0 and UNK to 3
        for i in range(len(source_numericalized)):
            inputs = np.array([bos] + source_numericalized[i] + [eos])
            output = inputs[1:]
            yield (inputs, output)

    def build(self,
              batch_size):

        my_gen = partial(MonolingualDataloaderWord._my_causal_lm_generator,
                         self._source_numericalized)

        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([None]),
                                                           tf.TensorShape([None])))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if self._vocab_size is not None:
            logger.debug("Vocab size limited to " + str(self._vocab_size))
            # Only to test performance with lower vocab size (and GPU mem)
            ds = ds.map(lambda x, y: (tf.minimum(x, self._vocab_size - 1),
                                      tf.minimum(y, self._vocab_size - 1)))

        ds = ds.padded_batch(batch_size=batch_size, padded_shapes=(self._seq_length, self._seq_length))

        self._build_all_dataset(ds, batch_size)
