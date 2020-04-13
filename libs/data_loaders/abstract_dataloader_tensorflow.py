from abc import ABC
from pathlib import Path
from typing import List

import tensorflow as tf
import tqdm
from tensorflow_datasets.core.features.text import SubwordTextEncoder

from libs.data_loaders.abstract_dataloader import AbstractSubwordTokenizer

logger = tf.get_logger()


class AbstractTensorFlowTokenizer(AbstractSubwordTokenizer, ABC):

    def __init__(self, config: dict, raw_english_test_set_file_path: str):

        AbstractSubwordTokenizer.__init__(self, config=config,
                                          raw_english_test_set_file_path=raw_english_test_set_file_path)

    def _load_tokenizer(self,
                        language: str,
                        tokenizer_algorithm: str,
                        vocab_size: int,
                        pretrained_model_dir_path: str,
                        corpora_filenames: List[str],
                        corpus_filename: str):
        tokenizer_filename_prefix = self._get_tokenizer_filename_prefix(language=language,
                                                                        tokenizer_algorithm=tokenizer_algorithm,
                                                                        vocab_size=vocab_size,
                                                                        dropout=0)
        logger.info(f"Specified tokenizer for lang {language}: {tokenizer_filename_prefix}")

        if (Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab.subwords")).exists():
            logger.info("Found an existing pretrained tokenizer for lang {language}: Load it")
            filepath = Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab")
            tokenizer: SubwordTextEncoder = SubwordTextEncoder.load_from_file(str(filepath))

        else:

            logger.info(f"No existing pretrained tokenizer for lang {language}: Train it")

            def gen_corpus():
                for filename in corpora_filenames:
                    text = self._read_file(corpus_filepath=self._folder / filename)
                    for line in tqdm.tqdm(text, "Build the tokenizer"):
                        yield line

            tokenizer: SubwordTextEncoder = SubwordTextEncoder.build_from_corpus(corpus_generator=gen_corpus(),
                                                                                 target_vocab_size=vocab_size,
                                                                                 reserved_tokens=self._special_tokens)
            # TF add its own extension
            filepath = str(Path(pretrained_model_dir_path) / (tokenizer_filename_prefix + "-vocab"))
            tokenizer.save_to_file(filepath)
            logger.info(f"Save tokenizer in {str(filepath)}")

        logger.info(f"Load dataset for lang {language}")
        corpus = self._read_file(corpus_filepath=self._folder / corpus_filename)

        logger.debug(f"Samples size: {len(corpus)} for tokenizer: {tokenizer_filename_prefix}")

        corpus = [tokenizer.encode(s) for s in corpus]

        return tokenizer, corpus

    def _decode(self, tokens, tokenizer: SubwordTextEncoder):
        pred: str = tokenizer.decode(tokens)
        eos_pos = pred.find(self._eos)
        if eos_pos != -1:
            return pred[:eos_pos]
        return pred
