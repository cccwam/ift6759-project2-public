import gc
import tempfile
from abc import ABC
from pathlib import Path
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractSubwordTokenizer
from libs.helpers import import_from

logger = tf.get_logger()


class AbstractHuggingFaceTokenizer(AbstractSubwordTokenizer, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractSubwordTokenizer.__init__(self, config=config,
                                          raw_english_test_set_file_path=raw_english_test_set_file_path)

    def _load_tokenizer(self,
                        language: str,
                        tokenizer_algorithm: str,
                        vocab_size: int,
                        max_seq_length: int,
                        dropout: float,
                        pretrained_model_dir_path: str,
                        corpora_filenames: List[str],
                        corpus_filename: str,
                        is_training: bool):
        tokenizer_filename_prefix = self._get_tokenizer_filename_prefix(language=language,
                                                                        tokenizer_algorithm=tokenizer_algorithm,
                                                                        vocab_size=vocab_size,
                                                                        dropout=dropout)
        logger.info(f"Specified tokenizer for lang {language}: {tokenizer_filename_prefix}")

        if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")).exists():
            filepath = Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.json")
        else:
            if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.txt")).exists():
                filepath = Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.txt")
            else:
                filepath = None

        if filepath:

            logger.info("Found an existing pretrained tokenizer for lang {language}: Load it")

            if dropout is None:
                if tokenizer_algorithm == "BertWordPieceTokenizer":
                    raise Exception("Bert WordPiece Tokenizer is not recommanded " +
                                    "for machine translation because of UNK tokens")
                    # tokenizer: BaseTokenizer = BertWordPieceTokenizer(vocab_file=str(filepath),
                    #                                                   unk_token=self._unk,
                    #                                                   sep_token=self._sep,
                    #                                                   cls_token=self._cls,
                    #                                                   clean_text=False,
                    #                                                   strip_accents=False,  # Changing this create UNK
                    #                                                   lowercase=False  # Changing this create UNK
                    #                                                   )
                else:
                    tokenizer: BaseTokenizer = import_from(
                        "tokenizers",
                        tokenizer_algorithm
                    )(vocab_file=str(filepath),
                      merges_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-merges.txt")))
            else:
                if tokenizer_algorithm == "BertWordPieceTokenizer":
                    # tokenizer: BaseTokenizer = BertWordPieceTokenizer(vocab_file=str(filepath),
                    #                                                   unk_token=self._unk,
                    #                                                   sep_token=self._sep,
                    #                                                   cls_token=self._cls,
                    #                                                   clean_text=False,
                    #                                                   strip_accents=False,  # Changing this create UNK
                    #                                                   lowercase=False  # Changing this create UNK
                    #                                                   )
                    raise Exception("Bert WordPiece Tokenizer is not recommanded " +
                                    "for machine translation because of UNK tokens")

                else:
                    tokenizer: BaseTokenizer = import_from(
                        "tokenizers",
                        tokenizer_algorithm
                    )(dropout=dropout if is_training and dropout != 0 else None,
                      vocab_file=str(filepath),
                      merges_file=str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-merges.txt")))

            tokenizer.add_special_tokens(self._special_tokens)

        else:

            logger.info(f"No existing pretrained tokenizer for lang {language}: Train it")

            if dropout is None:
                tokenizer: BaseTokenizer = import_from(
                    "tokenizers",
                    tokenizer_algorithm
                )()
            else:
                tokenizer: BaseTokenizer = import_from(
                    "tokenizers",
                    tokenizer_algorithm
                )(dropout=dropout if is_training and dropout != 0 else None)

            tokenizer.add_special_tokens(self._special_tokens)

            tokenizer = self._train_and_save(tokenizer=tokenizer,
                                             corpora_filenames=corpora_filenames,
                                             tokenizer_filename_prefix=tokenizer_filename_prefix,
                                             pretrained_model_dir_path=pretrained_model_dir_path,
                                             vocab_size=vocab_size,
                                             max_seq_length=max_seq_length)

        logger.info(f"Load dataset for lang {language}")

        corpus = self._read_file(corpus_filepath=self._folder / corpus_filename)

        logger.debug(f"Samples size: {len(corpus)} for tokenizer: {tokenizer_filename_prefix}")

        if is_training:
            # Tokenizers will be online for training to keep stochastic property
            return tokenizer, corpus

        # For inference, the tokenizer will be deterministic so we can load everything
        return tokenizer, tokenizer.encode_batch(corpus)

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
            corpora += self._read_file(corpus_filepath=self._folder / corpus_filename)

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

    def _decode(self, tokens: List[int], tokenizer: BaseTokenizer) -> str:
        pred: str = tokenizer.decode(tokens, skip_special_tokens=False)
        pred = pred.replace(self._bos, "")
        eos_pos = pred.find(self._eos)
        if eos_pos != -1:
            return pred[:eos_pos]
        return pred
