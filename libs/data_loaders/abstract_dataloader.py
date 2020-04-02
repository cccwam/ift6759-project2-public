import logging
from abc import abstractmethod, ABC
from functools import partial
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


class AbstractDataloader:

    # TODO: Implement the usage of raw_english_test_set_file_path. See TODO in evaluator.py's generate_predictions()
    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        """
        AbstractDataloader

        :param config: The configuration dictionary. It must follow configs/user/schema.json
        """
        self._dl_hparams = config["data_loader"]["hyper_params"]

        self._preprocessed_data_path = self._dl_hparams["preprocessed_data_path"]

        self._samples_for_test: int = self._dl_hparams["samples_for_test"]
        self._samples_for_valid: int = self._dl_hparams["samples_for_valid"]
        self._samples_for_train: int = self._dl_hparams["samples_for_train"]

        # To be initialized in build method (because called for each experiment during hparams search)
        self.test_dataset: Optional[tf.data.Dataset] = None
        self.valid_dataset: Optional[tf.data.Dataset] = None
        self.training_dataset: Optional[tf.data.Dataset] = None

    @abstractmethod
    def build(self,
              batch_size):
        raise NotImplementedError()

    @abstractmethod
    def get_hparams(self):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, tokens):
        raise NotImplementedError

    def _build_all_dataset(self, ds: tf.data.Dataset, ds_without_modification: tf.data.Dataset, batch_size: int):
        ds_without_modification = ds_without_modification.skip(int(self._samples_for_test / batch_size))
        self.valid_dataset_for_callbacks = ds_without_modification.take(int(self._samples_for_valid / batch_size))

        self.test_dataset = ds.take(int(self._samples_for_test / batch_size))
        ds = ds.skip(int(self._samples_for_test / batch_size))
        self.valid_dataset = ds.take(int(self._samples_for_valid / batch_size))
        self.training_dataset = ds.skip(int(self._samples_for_valid / batch_size))

        self._validation_steps = int(self._samples_for_valid / batch_size)

        if self._samples_for_train > 0:
            # Allow training with less samples and keeping the same validation size
            # Useful to train models more quickly
            self.training_dataset = self.training_dataset.take(int(self._samples_for_train / batch_size))

    # TF bug https://github.com/tensorflow/tensorflow/issues/28782
    #        self.test_dataset = self.test_dataset.cache()
    #        self.training_dataset = self.training_dataset.cache()
    #        self.valid_dataset = self.valid_dataset.cache()

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

    @abstractmethod
    def _my_generator(self, source_numericalized):
        raise NotImplementedError

    def _hook_dataset_post_precessing(self, ds):
        return ds

    def build(self,
              batch_size):
        my_gen = partial(self._my_generator,
                         source_numericalized=self._source_numericalized)

        assert self._output_types is not None, "Missing output_types"
        assert self._output_shapes is not None, "Missing _output_shapes"
        assert self._padded_shapes is not None, "Missing _padded_shapes"

        ds_without_modification = tf.data.Dataset.from_generator(my_gen,
                                            output_types=self._output_types,
                                            output_shapes=self._output_shapes)
        ds = self._hook_dataset_post_precessing(ds=ds_without_modification)

        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        ds_without_modification = ds_without_modification.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds_without_modification = ds_without_modification.prefetch(tf.data.experimental.AUTOTUNE)

        self._build_all_dataset(ds, ds_without_modification, batch_size)


class AbstractMonolingualCausalLMDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = (tf.float32, tf.float32) # TODO check if it works with int32
        self._output_shapes = (tf.TensorShape([None]),
                               tf.TensorShape([None]))
        self._padded_shapes = (self._seq_length, self._seq_length)


class AbstractMonolingualTransformersLMDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = (tf.float32,)
        self._output_shapes = (tf.TensorShape([None]),)
        self._padded_shapes = (self._seq_length,)


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

        self._en_numericalized = None
        self._fr_numericalized = None

    def _hook_dataset_post_precessing(self, ds):
        return ds

    @abstractmethod
    def _my_generator(self, source_numericalized, target_numericalized):
        raise NotImplementedError

    def build(self,
              batch_size):

        my_gen = partial(self._my_generator,
                         self._en_numericalized,
                         self._fr_numericalized)

        assert self._output_types is not None, "Missing output_types"
        assert self._output_shapes is not None, "Missing _output_shapes"
        assert self._padded_shapes is not None, "Missing _padded_shapes"

        ds_without_modification = tf.data.Dataset.from_generator(my_gen,
                                            output_types=self._output_types,
                                            output_shapes=self._output_shapes)
        ds = self._hook_dataset_post_precessing(ds=ds_without_modification)

        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        ds_without_modification = ds_without_modification.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds_without_modification = ds_without_modification.prefetch(tf.data.experimental.AUTOTUNE)

        self._build_all_dataset(ds, ds_without_modification, batch_size)


class AbstractBilingualSeq2SeqDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = ((tf.float32, tf.float32), tf.float32)
        self._output_shapes = ((tf.TensorShape([None]), tf.TensorShape([None])),
                               tf.TensorShape([None]))
        self._padded_shapes = ((self._seq_length_source, self._seq_length_target),
                               self._seq_length_target)


class AbstractBilingualTransformersDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = (tf.float32, tf.float32)
        self._output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
        self._padded_shapes = (self._seq_length_source, self._seq_length_target)
