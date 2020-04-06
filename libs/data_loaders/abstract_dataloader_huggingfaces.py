import gc
import pickle
import tempfile
from abc import ABC
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders import AbstractDataloader
from libs.helpers import import_from

logger = tf.get_logger()


class AbstractHuggingFacesTokenizer(AbstractDataloader, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractDataloader.__init__(self, config=config,
                                    raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._pad = "<pad>"
        self._mask = "<mask>"
        self._bos = "<bos>"
        self._eos = "<eos>"
        self._unk = "<unk>"

        self._special_tokens = [self._pad, self._mask, self._bos, self._eos, self._unk]

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
                corpora += [self._add_bos_eos(s) for s in corpus]

        logger.info(f"Train")

        with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
            # Huggingfaces requires a file to train, so we need to save all sentences into a file
            tmp.writelines(corpora)
            tokenizer.train(files=[tmp.name], show_progress=True, vocab_size=vocab_size,
                            special_tokens=self._special_tokens)

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

    @staticmethod
    def _decode(tokens, tokenizer: BaseTokenizer):
        return tokenizer.decode(tokens)

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

        if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")).exists():

            logger.info("Found an existing pretrained tokenizer for lang {language}: Load it")

            tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout,
              vocab_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")),
              merges_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-merges.txt")))

            tokenizer.add_special_tokens(self._special_tokens)

        else:

            logger.info(f"No existing pretrained tokenizer for lang {language}: Train it")

            tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout)

            tokenizer.add_special_tokens(self._special_tokens)

            tokenizer = self._train_and_save(tokenizer=tokenizer,
                                             corpora_filenames=corpora_filenames,
                                             tokenizer_filename_prefix=tokenizer_filename_prefix,
                                             pretrained_model_dir_path=pretrained_model_dir_path,
                                             vocab_size=vocab_size,
                                             max_seq_length=max_seq_length)

        logger.info(f"Load dataset for lang {language}")

        with open(str(self._folder / corpus_filename), 'rb') as handle:
            corpus = pickle.load(handle)
            corpus = [self._add_bos_eos(s) for s in corpus]

        corpus_numericalized = tokenizer.encode_batch(corpus)

        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(corpus_numericalized)}")
        return tokenizer, corpus_numericalized

    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float):
        return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

    def _add_bos_eos(self, s: List[str]):
        return self._bos + " ".join(s) + self._eos

    @staticmethod
    def _apply_mask_for_mlm(ds: tf.data.Dataset,
                            vocab_size: int,
                            with_multi_inputs=True):
        """
            Apply mask for masked language model

        Args:
            ds: dataset
            with_multi_inputs:

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

        # Inspiration from https://www.tensorflow.org/guide/data#applying_arbitrary_python_logic
        def _apply_mask_eager(inputs, output):
            input_shape = tf.shape(inputs)  # Shape Seq Length
            output_shape = tf.shape(output)  # Shape Seq Length

            # TODO set seed
            masks = distrib_mask.sample(input_shape,
                                        seed=42)  # Shape Seq Length * Probability for each class (4)
            masks = tf.cast(masks, dtype=tf.int32)
            random_tokens = distrib_random.sample(input_shape, seed=42)  # TODO set seed
            random_tokens = tf.cast(random_tokens, dtype=tf.int32)

            # Replace with mask
            # One is the mask token id
            inputs_masked = tf.where(tf.math.equal(masks[:, 2], 1), inputs, tf.ones(input_shape, dtype=tf.int32))

            # Replace with random token
            inputs_masked = tf.where(tf.math.equal(masks[:, 1], 1), inputs_masked, random_tokens)

            output_masked = tf.where(tf.math.equal(masks[:, 3], 1), output, tf.ones(output_shape, dtype=tf.int32))

            return inputs_masked, output_masked

        def apply_mask(x, y):
            inputs, attention_masks, tokens_type_ids = x
            inputs_masked, y = tf.py_function(_apply_mask_eager, [inputs, y], [tf.int32, tf.int32])

            return (inputs_masked, attention_masks, tokens_type_ids), y

        def apply_mask_single_input(x, y):
            inputs_masked, y = tf.py_function(_apply_mask_eager, [x, y], [tf.int32, tf.int32])

            return inputs_masked, y

        if with_multi_inputs:
            return ds.map(map_func=apply_mask)
        else:
            return ds.map(map_func=apply_mask_single_input)
