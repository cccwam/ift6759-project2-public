import logging
import pickle
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf

from pathlib import Path
import tempfile

from pip._internal.vcs import mercurial

from libs.data_loaders.abstract_dataloader import AbstractMonolingualDataloader
from tokenizers import (ByteLevelBPETokenizer,
                        CharBPETokenizer,
                        SentencePieceBPETokenizer,
                        BertWordPieceTokenizer, Encoding)

from libs.helpers import import_from

logger = logging.getLogger(__name__)


class MonolingualDataloaderSubword(AbstractMonolingualDataloader):
    """
        Dataset for monolingual corpora at subword level.
        This dataloader is using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict):

        super(MonolingualDataloaderSubword, self).__init__(config=config)

        pretrained_model_dir_path: str = self._dl_hparams["pretrained_model_dir_path"]
        assert pretrained_model_dir_path is not None, "Missing pretrained_model_dir_path in config"

        tokenizer_algorithm: str = self._dl_hparams["tokenizer_algorithm"]
        assert tokenizer_algorithm is not None, "Missing tokenizer_algorithm in config"

        language: str = self._dl_hparams["language"]
        assert language is not None, "Missing language in config"

        dropout: float = self._dl_hparams["dropout"]
        assert dropout is not None, "Missing dropout in config"

        tokenizer_filename_prefix = self._get_tokenizer_filename_prefix(language=language,
                                                                        tokenizer_algorithm=tokenizer_algorithm,
                                                                        vocab_size=self._vocab_size,
                                                                        dropout=dropout)
        logger.info(f"Specified tokenizer: {tokenizer_filename_prefix}")

        if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")).exists():

            logger.info("Found an existing pretrained tokenizer: Load it")

            self._tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout,
              vocab_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")),
              merges_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-merges.txt")))
        else:

            logger.info("No existing pretrained tokenizer: Train it")

            self._tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout)

            self._train_and_save(tokenizer_filename_prefix=tokenizer_filename_prefix,
                                 pretrained_model_dir_path=pretrained_model_dir_path)

        logger.info("Load monolingual dataset")

        monolingual_corpus_filename: str = self._dl_hparams["monolingual_corpus_filename"]
        assert monolingual_corpus_filename is not None, "Missing monolingual_corpus_filename in config"

        with open(self._preprocessed_data_path / monolingual_corpus_filename, 'rb') as handle:
            monolingual_corpus = pickle.load(handle)
            monolingual_corpus = [" ".join(s) for s in monolingual_corpus]

        self._source_numericalized = self._tokenizer.encode_batch(monolingual_corpus)

        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(self._source_numericalized)}")

    def _train_and_save(self, tokenizer_filename_prefix: str, pretrained_model_dir_path: str):
        corpora_filenames: List[str] = self._dl_hparams["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"

        tmp_path: str = self._dl_hparams["tmp_path"]
        assert tmp_path is not None, "Missing tmp_path in config"

        logger.info("Load all tokenized corpora")

        corpora = []
        for corpus_filename in corpora_filenames:
            with open(self._preprocessed_data_path / corpus_filename, 'rb') as handle:
                corpus = pickle.load(handle)
                corpora += [" ".join(l) for l in corpus]

        logger.info(f"Train")

        with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
            # Huggingfaces requires a file to train, so we need to save all sentences into a file
            tmp.writelines(corpora)
            self._tokenizer.train(files=[tmp.name], show_progress=True, vocab_size=self._vocab_size)

        logger.info("Compute the max length")

        tokenized_texts = self._tokenizer.encode_batch(corpora)
        seq_length = max([len(tokenized_text.ids) for tokenized_text in tokenized_texts])

        assert seq_length <= self._seq_length, ("ERROR: the maximum sequence length allowed in the dataloader " +
                                                "is lower than the maximum sequence length in corpora " +
                                                f"specified seq_length {self._seq_length} vs {seq_length}")

        logger.info(f"Max length: {seq_length}")

        logger.info("Save BPE tokenizer")

        self._tokenizer.save(pretrained_model_dir_path, tokenizer_filename_prefix)

    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float):
        return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

    def _my_causal_lm_generator(self, source_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            inputs = source_numericalized[i].ids
            output = inputs[1:]
            yield (inputs, output)
