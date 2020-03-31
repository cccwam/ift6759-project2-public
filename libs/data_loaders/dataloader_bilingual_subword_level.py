import logging
import pickle
import tempfile
from pathlib import Path
from typing import List

from tokenizers import (Encoding)
from tokenizers.implementations import BaseTokenizer

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader, AbstractBilingualSeq2SeqDataloader, \
    AbstractBilingualTransformersDataloader
from libs.helpers import import_from

logger = logging.getLogger(__name__)


class AbstractBilingualDataloaderSubword(AbstractBilingualDataloader):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractBilingualDataloader.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._folder: Path = Path(self._preprocessed_data_path["folder"])
        assert self._folder.exists()

        self._languages: List[str] = self._preprocessed_data_path["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        corpora_filenames: List[List[str]] = self._preprocessed_data_path["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"
        assert len(corpora_filenames) == 2, "You should have only two languages"

        pretrained_model_dir_path: str = self._dl_hparams["pretrained_model_dir_path"]
        assert pretrained_model_dir_path is not None, "Missing pretrained_model_dir_path in config"

        self._tokenizer_algorithm: str = self._dl_hparams["tokenizer_algorithm"]
        assert self._tokenizer_algorithm is not None, "Missing tokenizer_algorithm in config"

        self._dropout: float = self._dl_hparams["dropout"]
        assert self._dropout is not None, "Missing dropout in config"

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang1_en_tokenized.pickle")
        self._tokenizer_source, self._en_numericalized = res

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang2_fr_tokenized.pickle")
        self._tokenizer_source, self._fr_numericalized = res

    def _load_tokenizer(self,
                        language: str,
                        tokenizer_algorithm: str,
                        vocab_size: int,
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
        else:

            logger.info("No existing pretrained tokenizer for lang {language}: Train it")

            tokenizer: BaseTokenizer = import_from(
                "tokenizers",
                tokenizer_algorithm
            )(dropout=dropout)

            tokenizer = self._train_and_save(tokenizer=tokenizer,
                                             corpora_filenames=corpora_filenames,
                                             tokenizer_filename_prefix=tokenizer_filename_prefix,
                                             pretrained_model_dir_path=pretrained_model_dir_path)

        logger.info("Load dataset for lang {language}")

        with open(str(self._folder / corpus_filename), 'rb') as handle:
            corpus = pickle.load(handle)
            corpus = [" ".join(s) for s in corpus]

        corpus_numericalized = tokenizer.encode_batch(corpus)

        logger.debug(f"{str(self.__class__.__name__)} Samples: {len(corpus_numericalized)}")
        return tokenizer, corpus_numericalized

    def _train_and_save(self,
                        tokenizer: BaseTokenizer,
                        tokenizer_filename_prefix: str,
                        pretrained_model_dir_path: str,
                        corpora_filenames: List[str]):

        tmp_path: str = self._dl_hparams["tmp_path"]
        assert tmp_path is not None, "Missing tmp_path in config"

        logger.info("Load all tokenized corpora")

        corpora = []
        for corpus_filename in corpora_filenames:
            with open(str(self._folder / corpus_filename), 'rb') as handle:
                corpus = pickle.load(handle)
                corpora += [" ".join(l) for l in corpus]

        logger.info(f"Train")

        with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
            # Huggingfaces requires a file to train, so we need to save all sentences into a file
            tmp.writelines(corpora)
            tokenizer.train(files=[tmp.name], show_progress=True, vocab_size=self._vocab_size)

        logger.info("Compute the max length")

        tokenized_texts = self._tokenizer.encode_batch(corpora)
        seq_length = max([len(tokenized_text.ids) for tokenized_text in tokenized_texts])

        assert seq_length <= self._seq_length, ("ERROR: the maximum sequence length allowed in the dataloader " +
                                                "is lower than the maximum sequence length in corpora " +
                                                f"specified seq_length {self._seq_length} vs {seq_length}")

        logger.info(f"Max length: {seq_length}")

        logger.info("Save BPE tokenizer")

        tokenizer.save(pretrained_model_dir_path, tokenizer_filename_prefix)
        return tokenizer

    @staticmethod
    def _get_tokenizer_filename_prefix(language: str,
                                       tokenizer_algorithm: str,
                                       vocab_size: int,
                                       dropout: float):
        return f"{language}_{tokenizer_algorithm}_vocab_size_{vocab_size}_dropout_{dropout}"

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._languages[0],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_source,
                                                   dropout=self._dropout) + "-" + \
               self._get_tokenizer_filename_prefix(language=self._languages[1],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_target,
                                                   dropout=self._dropout)


class BilingualCausalLMDataloaderSubword(AbstractBilingualDataloaderSubword,
                                         AbstractBilingualSeq2SeqDataloader):
    """
        Dataset for bilingual corpora at subword level generating input sentence, target sentence
        and the shifted target sequence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualSeq2SeqDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = source_numericalized[i].ids
            target = target_numericalized[i].ids
            inputs = (source, target)
            output = target[1:]
            yield (inputs, output)


class BilingualTransformersDataloaderSubword(AbstractBilingualDataloaderSubword,
                                             AbstractBilingualTransformersDataloader):
    """
        Dataset for bilingual corpora at subword level generating only input sentence and target sentence

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractBilingualTransformersDataloader.__init__(self, config=config)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = source_numericalized[i].ids
            target = target_numericalized[i].ids
            yield source, target
