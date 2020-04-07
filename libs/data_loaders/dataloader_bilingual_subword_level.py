from abc import ABC
from typing import List

import numpy as np
import tensorflow as tf
from tokenizers import (Encoding)

from libs.data_loaders.abstract_dataloader import AbstractBilingualDataloader, AbstractBilingualSeq2SeqDataloader, \
    AbstractBilingualTransformersDataloader
from libs.data_loaders.abstract_dataloader_huggingfaces import AbstractHuggingFacesTokenizer

logger = tf.get_logger()


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

        self._languages: List[str] = self._preprocessed_data["languages"]
        assert self._languages is not None, "Missing languages in config"
        assert len(self._languages) == 2, "You should have only two languages"

        corpora_filenames: List[List[str]] = self._preprocessed_data["corpora_filenames"]
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



class BilingualTranslationEncoderDecoderDataloaderSubword(AbstractBilingualDataloaderSubword):
    """
    TODO
        Dataset for bilingual corpora at subword level generating input sentence, target sentence
        and masking the input for sentence 2.

    """

    def __init__(self, config: dict, raw_english_test_set_file_path: str):
        AbstractBilingualDataloaderSubword.__init__(self, config=config,
                                                    raw_english_test_set_file_path=raw_english_test_set_file_path)
        self._output_types = ((tf.int32, tf.int32, tf.float32, tf.float32, tf.float32),
                              tf.int32)
        self._output_shapes = ((tf.TensorShape([None]),
                                tf.TensorShape([None]),
                                tf.TensorShape([1, 1, self._seq_length_source]),
                                tf.TensorShape([1, self._seq_length_target, self._seq_length_target]),
                                tf.TensorShape([1, 1, self._seq_length_target])),
                               tf.TensorShape([None]))
        self._padded_shapes = ((self._seq_length_source, self._seq_length_source,
                                (1, 1, self._seq_length_source),
                                (1, self._seq_length_target, self._seq_length_target),
                                (1, 1, self._seq_length_target)),
                               self._seq_length_target)

    def _my_generator(self, source_numericalized: List[Encoding], target_numericalized: List[Encoding]):
        for i in range(len(source_numericalized)):
            source = np.zeros([self._seq_length_source], dtype=int)
            source[:len(source_numericalized[i].ids)] = source_numericalized[i].ids

            target = np.zeros([self._seq_length_target], dtype=int)
            target[0:len(target_numericalized[i].ids)] = target_numericalized[i].ids

            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(source, target)

            yield ((source, target, enc_padding_mask, combined_mask, dec_padding_mask), target)

    # Same as Blaise except that no batch size in dimension
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[tf.newaxis, tf.newaxis, :]  # (1, 1, seq_len)

    def create_look_ahead_mask(self, seq_length):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(self._seq_length_target)
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask
