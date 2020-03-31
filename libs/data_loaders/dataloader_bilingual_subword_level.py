import logging
from abc import ABC
from typing import List

from tokenizers import (Encoding)

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader, AbstractBilingualSeq2SeqDataloader, \
    AbstractBilingualTransformersDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = logging.getLogger(__name__)


class AbstractBilingualDataloaderSubword(AbstractBilingualDataloader, AbstractHuggingFacesTokenizer, ABC):
    """
        Abstract class containing most logic for dataset for bilingual corpora at subword level.
        It's using the Tokenizers library from HuggingFaces

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloader.__init__(self, config=config,
                                             raw_english_test_set_file_path=raw_english_test_set_file_path)
        AbstractHuggingFacesTokenizer.__init__(self, config=config,
                                               raw_english_test_set_file_path=raw_english_test_set_file_path)

        self._languages: List[str] = self._preprocessed_data_path["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        corpora_filenames: List[List[str]] = self._preprocessed_data_path["corpora_filenames"]
        assert corpora_filenames is not None, "Missing corpora_filenames in config"
        assert len(corpora_filenames) == 2, "You should have only two languages"

        res = self._load_tokenizer(language=self._languages[0],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_source,
                                   max_seq_length=self._seq_length_source,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[0],
                                   corpus_filename="train_lang1_en_tokenized.pickle")
        self._tokenizer_source, self._en_numericalized = res

        res = self._load_tokenizer(language=self._languages[1],
                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                   vocab_size=self._vocab_size_target,
                                   max_seq_length=self._seq_length_target,
                                   dropout=self._dropout,
                                   pretrained_model_dir_path=self._pretrained_model_dir_path,
                                   corpora_filenames=corpora_filenames[1],
                                   corpus_filename="train_lang2_fr_tokenized.pickle")
        self._tokenizer_target, self._fr_numericalized = res

    def get_hparams(self):
        return self._get_tokenizer_filename_prefix(language=self._languages[0],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_source,
                                                   dropout=self._dropout) + "-" + \
               self._get_tokenizer_filename_prefix(language=self._languages[1],
                                                   tokenizer_algorithm=self._tokenizer_algorithm,
                                                   vocab_size=self._vocab_size_target,
                                                   dropout=self._dropout)

    def decode(self, tokens: List[int]):
        return self._decode(tokens=tokens, tokenizer=self._tokenizer_target)


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
