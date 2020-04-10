from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional

import tensorflow as tf

logger = tf.get_logger()


class AbstractDataloader:

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        """
        AbstractDataloader

        :param config: The configuration dictionary. It must follow configs/user/schema.json
        """
        self._dl_hparams = config["data_loader"]["hyper_params"]

        self._preprocessed_data = self._dl_hparams["preprocessed_data"]

        self._samples_for_test: int = self._dl_hparams["samples_for_test"]
        self._samples_for_valid: int = self._dl_hparams["samples_for_valid"]
        self._samples_for_train: int = self._dl_hparams["samples_for_train"]

        # To be initialized in build method (because called for each experiment during hparams search)
        self.test_dataset: Optional[tf.data.Dataset] = None
        self.valid_dataset: Optional[tf.data.Dataset] = None
        self.training_dataset: Optional[tf.data.Dataset] = None
        self._validation_steps: Optional[int] = None

        self._raw_english_test_set_file_path: str = raw_english_test_set_file_path

    def build(self,
              batch_size):
        assert self._output_types is not None, "Missing output_types"
        assert self._output_shapes is not None, "Missing _output_shapes"
        assert self._padded_shapes is not None, "Missing _padded_shapes"

        ds_train = tf.data.Dataset.from_generator(self._get_train_generator(),
                                                  output_types=self._output_types,
                                                  output_shapes=self._output_shapes)

        ds_train = self._hook_dataset_post_precessing(ds=ds_train, is_training=True)
        ds_train = ds_train.padded_batch(batch_size=batch_size,
                                         padded_shapes=self._padded_shapes,
                                         drop_remainder=True)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.skip(int(self._samples_for_test / batch_size))  # Skip samples for test
        ds_train = ds_train.skip(int(self._samples_for_valid / batch_size))  # Skip samples for valid
        self.training_dataset = ds_train.skip(int(self._samples_for_valid / batch_size))

        ds_valid_test = tf.data.Dataset.from_generator(self._get_valid_generator(),
                                                       output_types=self._output_types,
                                                       output_shapes=self._output_shapes)
        ds_valid_test = self._hook_dataset_post_precessing(ds=ds_valid_test, is_training=False)
        ds_valid_test = ds_valid_test.padded_batch(batch_size=batch_size,
                                                   padded_shapes=self._padded_shapes,
                                                   drop_remainder=True)
        ds_valid_test = ds_valid_test.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_dataset = ds_valid_test.take(int(self._samples_for_test / batch_size))
        ds_valid_test = ds_valid_test.skip(int(self._samples_for_test / batch_size))
        self.valid_dataset = ds_valid_test.take(int(self._samples_for_valid / batch_size))

        if self._samples_for_train > 0:
            # Allow training with less samples and keeping the same validation size
            # Useful to train models more quickly
            self.training_dataset = self.training_dataset.take(int(self._samples_for_train / batch_size))

        self._validation_steps = int(self._samples_for_valid / batch_size)

    def _hook_dataset_post_precessing(self, ds: tf.data.Dataset, is_training: bool):
        return ds

    @abstractmethod
    def _get_train_generator(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_valid_generator(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_test_generator(self):
        raise NotImplementedError()

    @abstractmethod
    def get_hparams(self):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, tokens):
        raise NotImplementedError

    def build_test_dataset(self, batch_size: int):
        assert self._raw_english_test_set_file_path is not None, "Missing raw_english_test_set_file_path"
        test_input_filepath = Path(self._raw_english_test_set_file_path)
        assert test_input_filepath.exists(), "The test input file doesn't exist"

        assert self._output_test_types is not None, "Missing _output_test_types"
        assert self._output_test_shapes is not None, "Missing _output_test_shapes"
        assert self._padded_test_shapes is not None, "Missing _padded_test_shapes"

        ds = tf.data.Dataset.from_generator(self._get_test_generator(),
                                            output_types=self._output_test_types,
                                            output_shapes=self._output_test_shapes)
        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_test_shapes)
        self.test_dataset = ds

    @property
    def validation_steps(self):
        assert self._validation_steps is not None, "You must call build before"
        return self._validation_steps


class AbstractMonolingualDataloader(AbstractDataloader, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractDataloader.__init__(self, config=config,
                                    raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._vocab_size: int = self._dl_hparams["vocab_size"]
        assert self._vocab_size is not None, "vocab_size missing"
        self._seq_length: int = self._dl_hparams["seq_length"]
        assert self._seq_length is not None, "seq_length missing"

        self._output_types = None


class AbstractBilingualDataloader(AbstractDataloader, ABC):

    # noinspection PyUnusedLocal
    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        super(AbstractBilingualDataloader, self).__init__(config=config,
                                                          raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._vocab_size_source: int = self._dl_hparams["vocab_size_source"]
        assert self._vocab_size_source is not None, "vocab_size_source missing"
        self._seq_length_source: int = self._dl_hparams["seq_length_source"]
        assert self._seq_length_source is not None, "seq_length_source missing"

        self._vocab_size_target: int = self._dl_hparams["vocab_size_target"]
        assert self._vocab_size_target is not None, "vocab_size_target missing"
        self._seq_length_target: int = self._dl_hparams["seq_length_target"]
        assert self._seq_length_target is not None, "seq_length_target missing"

    def get_seq_length(self):
        return self._seq_length_target
