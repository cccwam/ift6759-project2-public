from abc import abstractmethod, ABC
from functools import partial
from pathlib import Path
from typing import Optional, List

import tensorflow as tf
from tokenizers.implementations import BaseTokenizer
import logging
import pickle
import tempfile
import gc

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

        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=self._output_types,
                                            output_shapes=self._output_shapes)
        ds = self._hook_dataset_post_precessing(ds=ds)

        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self._build_all_dataset(ds, batch_size)


class AbstractMonolingualCausalLMDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = (tf.float32, tf.float32)
        self._output_shapes = (tf.TensorShape([None]),
                               tf.TensorShape([None]))
        self._padded_shapes = (self._seq_length, self._seq_length)


class AbstractMonolingualTransformersLMDataloader:

    # noinspection PyUnusedLocal
    def __init__(self, config: dict):
        self._output_types = (tf.float32,)
        self._output_shapes = (tf.TensorShape([None]),)
        self._padded_shapes = (self._seq_length,)


class AbstractHuggingFacesTokenizer(AbstractDataloader, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractDataloader.__init__(self, config=config,
                                    raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._bos = "<BOS>"
        self._eos = "<EOS>"

        self._folder: Path = Path(self._preprocessed_data_path["folder"])
        assert self._folder.exists()

        self._pretrained_model_dir_path: str = self._dl_hparams["pretrained_model_dir_path"]
        assert self._pretrained_model_dir_path is not None, "Missing pretrained_model_dir_path in config"

        self._tokenizer_algorithm: str = self._dl_hparams["tokenizer_algorithm"]
        assert self._tokenizer_algorithm is not None, "Missing tokenizer_algorithm in config"

        self._dropout: float = self._dl_hparams["dropout"]
        assert self._dropout is not None, "Missing dropout in config"


    def _train_and_save(self,
                        tokenizer: BaseTokenizer,
                        tokenizer_filename_prefix: str,
                        pretrained_model_dir_path: str,
                        corpora_filenames: List[str],
                        vocab_size: int,
                        max_seq_length: int):

        tmp_path: str = self._dl_hparams["tmp_path"]
        assert tmp_path is not None, "Missing tmp_path in config"

        logger.info("Load all tokenized corpora")

        corpora = []
        for corpus_filename in corpora_filenames:
            with open(str(self._folder / corpus_filename), 'rb') as handle:
                corpus = pickle.load(handle)
                corpora += [self._bos + " ".join(s) + self._eos for s in corpus]

        logger.info(f"Train")

        with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
            # Huggingfaces requires a file to train, so we need to save all sentences into a file
            tmp.writelines(corpora)
            tokenizer.train(files=[tmp.name], show_progress=True, vocab_size=vocab_size)

        logger.info("Compute the max length")

        # Compute the max length by batch to avoid Killed message (appears to be OOM error)
        i, batch, seq_length = 0, 2 ** 16, 0
        while i < len(corpora):
            tokenized_texts = tokenizer.encode_batch(corpora[i:min(i + batch, len(corpora))])
            seq_length = max([len(tokenized_text.ids) for tokenized_text in tokenized_texts] + [seq_length])
            i += batch
            gc.collect()

        assert seq_length <= max_seq_length, (
                "ERROR: the maximum sequence length allowed in the dataloader " +
                "is lower than the maximum sequence length in corpora " +
                f"specified seq_length {seq_length} vs {max_seq_length}")

        logger.info(f"Max length: {seq_length}")

        logger.info("Save BPE tokenizer")

        tokenizer.save(pretrained_model_dir_path, tokenizer_filename_prefix)
        return tokenizer

    def _decode(self, tokens, tokenizer):
        return tokenizer.decode([t for t in tokens.astype(int) if t != 0])

    def _load_tokenizer(self,
                        language: str,
                        tokenizer_algorithm: str,
                        vocab_size: int,
                        max_seq_length: int,
                        dropout: float,
                        pretrained_model_dir_path: str,
                        corpora_filenames: List[str],
                        corpus_filename: str):
        tokenizer_filename_prefix = self._get_tokenizer_filename_prefix(language=language,
                                                                        tokenizer_algorithm=tokenizer_algorithm,
                                                                        vocab_size=vocab_size,
                                                                        dropout=dropout)
        logger.info(f"Specified tokenizer for lang {language}: {tokenizer_filename_prefix}")

        # Import from here to prevent circular imports
        from libs.helpers import import_from

        if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")).exists():

            logger.info("Found an existing pretrained tokenizer for lang {language}: Load it")

            tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout,
              vocab_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")),
              merges_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-merges.txt")))
        else:

            logger.info(f"No existing pretrained tokenizer for lang {language}: Train it")

            tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout)

            tokenizer.add_special_tokens([self._bos, self._eos])

            tokenizer = self._train_and_save(tokenizer=tokenizer,
                                             corpora_filenames=corpora_filenames,
                                             tokenizer_filename_prefix=tokenizer_filename_prefix,
                                             pretrained_model_dir_path=pretrained_model_dir_path,
                                             vocab_size=vocab_size,
                                             max_seq_length=max_seq_length)

        logger.info(f"Load dataset for lang {language}")

        with open(str(self._folder / corpus_filename), 'rb') as handle:
            corpus = pickle.load(handle)
            corpus = [self._bos + " ".join(s) + self._eos for s in corpus]

        corpus_numericalized = tokenizer.encode_batch(corpus)

        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(corpus_numericalized)}")
        return tokenizer, corpus_numericalized


    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float):
        return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

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

        ds = tf.data.Dataset.from_generator(my_gen,
                                            output_types=self._output_types,
                                            output_shapes=self._output_shapes)
        ds = self._hook_dataset_post_precessing(ds=ds)

        ds = ds.padded_batch(batch_size=batch_size,
                             padded_shapes=self._padded_shapes,
                             drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self._build_all_dataset(ds, batch_size)


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
