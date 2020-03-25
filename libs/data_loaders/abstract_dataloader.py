from abc import abstractmethod
from pathlib import Path
from typing import Optional

import tensorflow as tf


class AbstractDataloader:

    def __init__(self, config: dict):
        self._dl_hparams = config["data_loader"]["hyper_params"]

        self._preprocessed_data_path = Path(self._dl_hparams["preprocessed_data_path"])

        self._samples_for_test: int = self._dl_hparams["samples_for_test"]
        self._samples_for_valid: int = self._dl_hparams["samples_for_valid"]
        self._samples_for_train: int = self._dl_hparams["samples_for_train"]

        self._vocab_size: int = self._dl_hparams["vocab_size"]
        self._seq_length: int = self._dl_hparams["seq_length"]

        # To be initialized in build method (because called for each experiment during hparams search)
        self.test_dataset: Optional[tf.data.Dataset] = None
        self.valid_dataset: Optional[tf.data.Dataset] = None
        self.training_dataset: Optional[tf.data.Dataset] = None

    @abstractmethod
    def build(self,
              batch_size):
        raise NotImplementedError()

    # TODO is it needed ?
    @property
    @abstractmethod
    def get_token_to_word(self):
        raise NotImplementedError()

    # TODO is it needed ?
    def decode(self, tokens):
        mapped_tokens = [self.get_token_to_word[t] for t in tokens]
        return " ".join(mapped_tokens)

    def _build_all_dataset(self, ds: tf.data.Dataset, batch_size: int):
        self.test_dataset = ds.take(int(self._samples_for_test / batch_size))
        ds = ds.skip(int(self._samples_for_test / batch_size))
        self.valid_dataset = ds.take(int(self._samples_for_valid / batch_size))
        self.training_dataset = ds.skip(int(self._samples_for_valid / batch_size))

        self._validation_steps = int(self._samples_for_valid / batch_size)

        if self._samples_for_train > 0:
            # Allow training with less samples and keeping the same validation size
            # Useful to train models more quickly
            self.training_dataset = self.training_dataset.take(int(self._samples_for_train / batch_size))

    @property
    def validation_steps(self):
        assert self._validation_steps is not None, "You must call build before"
        return self._validation_steps
