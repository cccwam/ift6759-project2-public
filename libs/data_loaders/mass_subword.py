import os

import tensorflow as tf

from libs.data_loaders import subwords
from libs.models import transformer

TextLineDataset = tf.data.TextLineDataset


def create_for_transformer_pre_mask(tokenizer):
    def encode(lang1, lang2):
        """Encode with tokenizer and add beginning/end of sentence tokens

        :param lang1: sentence
        :param lang2: sentence
        :return: list of tokens for both languages
        """
        lang1 = [tokenizer.vocab_size] + tokenizer.encode(
            lang1.numpy()) + [tokenizer.vocab_size + 1]

        lang2_in = [tokenizer.vocab_size] + tokenizer.encode(
            lang2.numpy())

        lang2_out = tokenizer.encode(
            lang2.numpy()) + [tokenizer.vocab_size + 1]

        return lang1, lang2_in, lang2_out

    def tf_py_encode(lang1, lang2):
        result_lang1, result_lang2, result_lang2_out = tf.py_function(
            encode, [lang1, lang2], [tf.int64, tf.int64, tf.int64])
        result_lang1.set_shape([None])
        result_lang2.set_shape([None])
        result_lang2_out.set_shape([None])

        return result_lang1, result_lang2, result_lang2_out

    return tf_py_encode


class MassSubwordDataLoader:
    """
        Dataset for bilingual corpora at subword level.

        Special tokens:
        - 0 for MASK
        - vocab_size for BOS
        - vocab_size + 1 for EOS
    """

    def __init__(self, config: dict, raw_english_test_set_file_path=None,
                 **kwargs):
        self.config = config
        self.training_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        self.vocab_size_source = None
        self.vocab_size_target = None
        self.validation_steps = None
        self.raw_english_test_set_file_path = raw_english_test_set_file_path

    def build(self, batch_size, mode='translate'):
        dl_hparams = self.config["data_loader"]["hyper_params"]
        path_data = dl_hparams["preprocessed_data_path"]["folder"]
        vocabulary_name = dl_hparams["vocabulary_name"]

        path_trad_en_train = os.path.join(path_data, 'original',
                                          'train.lang1.train')
        path_trad_en_test = os.path.join(path_data, 'original',
                                         'train.lang1.test')
        path_unal_en_train = os.path.join(path_data,
                                          'tokenized_default_no_punc',
                                          'unaligned.en.train')
        path_trad_en_val = os.path.join(path_data, 'original',
                                        'train.lang1.validation')
        path_unal_en_val = os.path.join(path_data, 'tokenized_default_no_punc',
                                        'unaligned.en.validation')
        path_trad_fr_train = os.path.join(path_data, 'original',
                                          'train.lang2.train')
        path_trad_fr_test = os.path.join(path_data, 'original',
                                         'train.lang2.test')
        path_unal_fr_train = os.path.join(path_data, 'tokenized_keep_case',
                                          'unaligned.fr.train')
        path_trad_fr_val = os.path.join(path_data, 'original',
                                        'train.lang2.validation')
        path_unal_fr_val = os.path.join(path_data, 'tokenized_keep_case',
                                        'unaligned.fr.validation')
        # Create all TextLineDataset objects
        datasets = {
            'sentences_translation_en_train': TextLineDataset(
                [path_trad_en_train]),
            'sentences_translation_en_validation': TextLineDataset(
                [path_trad_en_val]),
            'sentences_translation_en_test': TextLineDataset(
                [path_trad_en_test]),
            'sentences_all_train': TextLineDataset(
                [path_trad_en_train, path_unal_en_train,
                 path_trad_fr_train, path_unal_fr_train]),
            'sentences_all_validation': TextLineDataset(
                [path_trad_en_val, path_unal_en_val,
                 path_trad_fr_val, path_unal_fr_val]),
            'sentences_translation_fr_train': TextLineDataset(
                [path_trad_fr_train]),
            'sentences_translation_fr_validation': TextLineDataset(
                [path_trad_fr_val]),
            'sentences_translation_fr_test': TextLineDataset(
                [path_trad_fr_test]),
        }
        if self.raw_english_test_set_file_path:
            self.test_dataset = TextLineDataset([self.raw_english_test_set_file_path])

        self.tokenizer = subwords.subword_tokenizer(
            vocabulary_name, datasets['sentences_all_train'])
        self.vocab_size_source = self.tokenizer.vocab_size + 2
        print(self.vocab_size_source)

        if mode == 'all':
            # ToDo use the whole dataset
            sentences_translation_both_train = tf.data.Dataset.zip(
                (datasets['sentences_all_en_train'].take(30000),
                 datasets['sentences_all_en_train'].take(30000)))
            sentences_translation_both_validation = tf.data.Dataset.zip(
                (datasets['sentences_all_en_validation'].take(5000),
                 datasets['sentences_all_en_validation'].take(5000)))
            # sentences_translation_both_train = tf.data.Dataset.zip(
            #     (datasets['sentences_all_en_train'],
            #      datasets['sentences_all_en_train']))
            # sentences_translation_both_validation = tf.data.Dataset.zip(
            #     (datasets['sentences_all_en_validation'],
            #      datasets['sentences_all_en_validation']))
        else:
            sentences_translation_both_train = tf.data.Dataset.zip(
                (datasets['sentences_translation_en_train'],
                 datasets['sentences_translation_fr_train']))
            sentences_translation_both_validation = tf.data.Dataset.zip(
                (datasets['sentences_translation_en_validation'],
                 datasets['sentences_translation_fr_validation']))

        tf_encode = create_for_transformer_pre_mask(self.tokenizer)

        train_preprocessed = (
            sentences_translation_both_train.map(tf_encode))
        train_padded = train_preprocessed.padded_batch(
            batch_size, padded_shapes=([None], [None], [None])).prefetch(tf.data.experimental.AUTOTUNE).cache()

        # Hack for tf 2.0.0, need to cache train_padded before creating the generator
        for _ in train_padded:
            pass

        def mass_generator_train():
            for enc_inp, dec_inp, dec_out in train_padded:
                enc_padding_mask, combined_mask, _ = transformer.create_masks(enc_inp, dec_inp)
                yield (enc_inp, dec_inp, enc_padding_mask, combined_mask), dec_out

        self.training_dataset = tf.data.Dataset.from_generator(
            mass_generator_train, ((tf.int64, tf.int64, tf.float32, tf.float32), tf.int64),
            output_shapes=(
                ((tf.TensorShape([None, None]),
                  tf.TensorShape([None, None]),
                  tf.TensorShape([None, 1, 1, None]),
                  tf.TensorShape([None, 1, None, None])),
                 tf.TensorShape([None, None])))
        )

        val_preprocessed = (
            sentences_translation_both_validation.map(tf_encode))
        val_padded = val_preprocessed.padded_batch(
            batch_size, padded_shapes=([None], [None], [None])).prefetch(tf.data.experimental.AUTOTUNE).cache()

        # Hack for tf 2.0.0, need to cache val_padded before creating the generator
        for _ in val_padded:
            pass

        def mass_generator_val():
            for enc_inp, dec_inp, dec_out in val_padded:
                enc_padding_mask, combined_mask, _ = transformer.create_masks(enc_inp, dec_inp)
                yield (enc_inp, dec_inp, enc_padding_mask, combined_mask), dec_out

        self.valid_dataset = tf.data.Dataset.from_generator(
            mass_generator_val, ((tf.int64, tf.int64, tf.float32, tf.float32), tf.int64),
            output_shapes=(
                ((tf.TensorShape([None, None]),
                  tf.TensorShape([None, None]),
                  tf.TensorShape([None, 1, 1, None]),
                  tf.TensorShape([None, 1, None, None])),
                 tf.TensorShape([None, None])))
        )

    def get_hparams(self):
        return f"vocab_size_{self.vocab_size_source},{self.vocab_size_target}"
