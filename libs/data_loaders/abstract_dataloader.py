import pickle
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
        self._train_steps: Optional[int] = None
        self._batch_size: Optional[int] = None

        # TODO should we have one seed in cli or config ?
        self._seed: int = self._dl_hparams["seed"] if "seed" in self._dl_hparams else 42

        self._raw_english_test_set_file_path: str = raw_english_test_set_file_path

    def build(self,
              batch_size):
        self._batch_size = batch_size

        self.training_dataset = self._get_train_dataset(batch_size=batch_size)

        # Skip valid and test set
        self.training_dataset = self.training_dataset.skip(np.ceil(self._samples_for_test / batch_size) +
                                                           np.ceil(self._samples_for_valid / batch_size))
        self.training_dataset = self.training_dataset

        ds_valid_test = self._get_valid_dataset(batch_size=batch_size)

        if self._samples_for_test != 0:
            self.test_dataset = ds_valid_test.take(np.ceil(self._samples_for_test / batch_size))
            ds_valid_test = ds_valid_test.skip(np.ceil(self._samples_for_test / batch_size))
        ds_valid_test = ds_valid_test.take(np.ceil(self._samples_for_valid / batch_size))
        self.valid_dataset = ds_valid_test  # .cache() # Cache will not work because of BLEU callback

        if self._samples_for_train > 0:
            # Allow training with less samples and keeping the same validation size
            # Useful to train models more quickly
            self.training_dataset = self.training_dataset.take(np.ceil(self._samples_for_train / batch_size))
            if self._samples_for_train < 12000:  # Prevent OOM
                self.training_dataset = self.training_dataset.cache()
            self.training_dataset = self.training_dataset.repeat()
        else:
            if self._samples_for_train < 12000:  # Prevent OOM
                self.training_dataset = self.training_dataset.cache()

        if self._samples_for_train != -1:
            self._train_steps = np.ceil(self._samples_for_train / batch_size)
        else:
            self._train_steps = None
        self._validation_steps = np.ceil(self._samples_for_valid / batch_size)

        if self._raw_english_test_set_file_path is not None:
            self.build_test_dataset(batch_size=batch_size)

    @abstractmethod
    def _get_train_dataset(self, batch_size: int) -> tf.data.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def _get_valid_dataset(self, batch_size: int) -> tf.data.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def _get_test_dataset(self, batch_size: int) -> tf.data.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def get_hparams(self):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, tokens) -> str:
        raise NotImplementedError

    def build_test_dataset(self, batch_size: int):
        assert self._raw_english_test_set_file_path is not None, "Missing raw_english_test_set_file_path"
        test_input_filepath = Path(self._raw_english_test_set_file_path)
        assert test_input_filepath.exists(), "The test input file doesn't exist"

        self.test_dataset = self._get_test_dataset(batch_size=batch_size)

    @property
    def samples_for_valid(self) -> int:
        return self._samples_for_valid

    @property
    def validation_steps(self) -> int:
        assert self._validation_steps is not None, "You must call build before"
        return self._validation_steps

    @property
    def train_steps(self) -> Optional[int]:
        return self._train_steps  # This may be null

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size  # This may be null


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


class AbstractSubwordTokenizer(AbstractDataloader, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractDataloader.__init__(self, config=config,
                                    raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._pad = "<pad>"
        self._mask = "<mask>"
        self._bos = "<bos>"
        self._eos = "<eos>"
        self._unk = "<unk>"
        self._sep = "<sep>"
        self._cls = "<cls"

        self._special_tokens = [self._pad, self._mask, self._bos, self._eos, self._unk, self._sep, self._cls]

        self._folder: Path = Path(self._preprocessed_data["folder"])
        assert self._folder.exists()

        self._pretrained_model_dir_path: str = self._dl_hparams["pretrained_model_dir_path"]
        assert self._pretrained_model_dir_path is not None, "Missing pretrained_model_dir_path in config"

        self._tokenizer_algorithm: str = self._dl_hparams["tokenizer_algorithm"]
        assert self._tokenizer_algorithm is not None, "Missing tokenizer_algorithm in config"

        if "dropout" in self._dl_hparams:
            self._dropout: Optional[float] = self._dl_hparams["dropout"]  # Optional
        else:
            self._dropout: Optional[float] = None

    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float) -> str:
        if dropout is None:
            return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}"
        else:
            return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

    def _add_bos_eos_tokenized_sentence(self, s: List[str]) -> str:
        return self._add_bos_eos(" ".join(s))

    def _add_bos_eos(self, s: str) -> str:
        return self._bos + s + self._eos

    def _apply_mask_for_mlm(self,
                            ds: tf.data.Dataset,
                            vocab_size: int):
        """
            Apply mask for masked language model

        Args:
            ds: dataset
            vocab_size: vocab size

        Returns:

        """

        # Do action only for 15% of tokens (and mask output for others)
        prob_mask_idx = 0.15
        # 10% nothing to do, 10% random word, 80% mask
        # prob_nothing, prob_random_replacement, prob_replace_by_mask \
        prob_mask_actions = np.array([0.1, 0.1, 0.8])
        prob_mask_actions = prob_mask_actions * prob_mask_idx
        prob_mask_actions = np.append(prob_mask_actions, [1 - sum(prob_mask_actions)]).tolist()
        distrib_mask = tfp.distributions.Multinomial(total_count=1,
                                                     probs=prob_mask_actions)

        distrib_random = tfp.distributions.Uniform(low=len(self._special_tokens), high=vocab_size)

        def apply_mask(x, output):
            inputs, enc_padding_mask = x

            input_shape = tf.shape(inputs)  # Batch size * Seq Length
            output_shape = tf.shape(output)  # Batch size * Seq Length

            masks = distrib_mask.sample(input_shape,
                                        seed=self._seed)  # Batch size *Seq Length * Probability for each class (4)
            masks = tf.cast(masks, dtype=tf.int32)
            random_tokens = distrib_random.sample(input_shape, seed=self._seed)
            random_tokens = tf.cast(random_tokens, dtype=tf.int32)

            # Replace with mask
            # One is the mask token id
            inputs_masked = tf.where(tf.math.equal(masks[:, :, 2], 1), inputs, tf.ones(input_shape, dtype=tf.int32))

            # Replace with random token
            inputs_masked = tf.where(tf.math.equal(masks[:, :, 1], 1), inputs_masked, random_tokens)

            output_masked = tf.where(tf.math.equal(masks[:, :, 3], 1), output, tf.ones(output_shape, dtype=tf.int32))

            return (inputs_masked, enc_padding_mask), output_masked

        return ds.map(map_func=apply_mask)

    @staticmethod
    def _create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def _create_look_ahead_mask(seq_length):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        return mask  # (seq_len, seq_len)

    @tf.function
    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self._create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self._create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self._create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def _read_file(self, corpus_filepath: Path) -> List[str]:
        if corpus_filepath.suffix == ".pickle":
            with open(str(corpus_filepath), 'rb') as handle:
                corpus = pickle.load(handle)
                return [self._add_bos_eos_tokenized_sentence(s) for s in corpus]
        else:
            with open(str(corpus_filepath), 'r') as handle:
                corpus = handle.read().split('\n')
                return [self._add_bos_eos(s) for s in corpus]

    @property
    @abstractmethod
    def bos(self) -> List[int]:
        raise NotImplementedError()  # TODO add for HF


class AbstractBilingualDataloaderSubword(AbstractBilingualDataloader, AbstractSubwordTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloader.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractSubwordTokenizer.__init__(self, config=config,
                                          raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._languages: List[str] = self._preprocessed_data["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        self._corpora_filenames: List[List[str]] = self._preprocessed_data["corpora_filenames"]
        assert self._corpora_filenames is not None, "Missing corpora_filenames in config"
        assert len(self._corpora_filenames) == 2, "You should have only two languages"

        self._bilingual_corpus_filenames: List[str] = self._preprocessed_data["bilingual_corpus_filenames"]
        assert self._bilingual_corpus_filenames is not None, "Missing bilingual_corpus_filenames in config"
        assert len(self._bilingual_corpus_filenames) == 2, "You should have only two languages"

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._languages[0],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_source,
                                                   dropout=self._dropout) + "-" + \
               self._get_tokenizer_filename_prefix(language=self._languages[1],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_target,
                                                   dropout=self._dropout)
