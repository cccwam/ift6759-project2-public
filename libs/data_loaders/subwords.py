import os

import tensorflow as tf
import tensorflow_datasets as tfds
TextLineDataset = tf.data.TextLineDataset


def subword_tokenizer(vocabulary_file, sentences, target_vocab_size=2**13,
                      force_compute=False):
    """Subword tokenizer for given sentences

    :param vocabulary_file: str, path to vocabulary file on disk
    :param sentences: TextLineDataset of the sentences
    :param target_vocab_size: int
    :param force_compute: force computation even if vocabulary_file exists
    :return: tokenizer
    """

    if force_compute or (not os.path.isfile(vocabulary_file + '.subwords')):
        print(f"Computing subword vocabulary ({vocabulary_file})")
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (sentence.numpy() for sentence in sentences),
            target_vocab_size=target_vocab_size)
        tokenizer.save_to_file(vocabulary_file)
    else:
        print(f"Loading precomputed subword vocabulary ({vocabulary_file})")
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            vocabulary_file)
    return tokenizer


def create_tf_encoder_bilingual(tokenizer1, tokenizer2):
    def encode(lang1, lang2):
        """Encode with tokenizer and add beginning/end of sentence tokens

        :param lang1: sentence
        :param lang2: sentence
        :return: list of tokens for both languages
        """
        lang1 = [tokenizer1.vocab_size] + tokenizer1.encode(
            lang1.numpy()) + [tokenizer1.vocab_size + 1]

        lang2 = [tokenizer2.vocab_size] + tokenizer2.encode(
            lang2.numpy()) + [tokenizer2.vocab_size + 1]

        return lang1, lang2

    def tf_py_encode(lang1, lang2):
        result_lang1, result_lang2 = tf.py_function(
            encode, [lang1, lang2], [tf.int64, tf.int64])
        result_lang1.set_shape([None])
        result_lang2.set_shape([None])

        return result_lang1, result_lang2

    return tf_py_encode


class SubwordDataLoader:
    """
        Dataset for bilingual corpora at subword level.

        Special tokens:
        - 0 for MASK
        - vocab_size for BOS
        - vocab_size + 1 for EOS
    """

    def __init__(self, config: dict):
        self.config = config
        self.training_dataset = None
        self.validation_dataset = None
        self.tokenizer_en = None
        self.tokenizer_fr = None
        self.input_vocab_size = None
        self.target_vocab_size = None
        self.validation_steps = 5

    def build(self, batch_size):
        dl_hparams = self.config["data_loader"]["hyper_params"]
        path_data = dl_hparams["preprocessed_data_path"]
        vocabulary_name_en = dl_hparams["vocabulary_name_en"]
        vocabulary_name_fr = dl_hparams["vocabulary_name_fr"]

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
        path_unal_fr_val = os.path.join(path_data, 'tokenized_keep_caseun'
                                                   '',
                                        'unaligned.fr.validation')
        # Create all TextLineDataset objects
        datasets = {
            'sentences_translation_en_train': TextLineDataset(
                [path_trad_en_train]),
            'sentences_translation_en_test': TextLineDataset(
                [path_trad_en_test]),
            'sentences_all_en_train': TextLineDataset(
                [path_trad_en_train, path_unal_en_train]),
            'sentences_translation_en_validation': TextLineDataset(
                [path_trad_en_val]),
            'sentences_all_en_validation': TextLineDataset(
                [path_trad_en_val, path_unal_en_val]),
            'sentences_translation_fr_train': TextLineDataset(
                [path_trad_fr_train]),
            'sentences_translation_fr_test': TextLineDataset(
                [path_trad_fr_test]),
            'sentences_all_fr_train': TextLineDataset(
                [path_trad_fr_train, path_unal_fr_train]),
            'sentences_translation_fr_validation': TextLineDataset(
                [path_trad_fr_val]),
            'sentences_all_fr_validation': TextLineDataset(
                [path_trad_fr_val, path_unal_fr_val]),
        }

        self.tokenizer_en = subword_tokenizer(
            vocabulary_name_en, datasets['sentences_translation_en_train'])
        self.input_vocab_size = self.tokenizer_en.vocab_size + 2
        self.tokenizer_fr = subword_tokenizer(
            vocabulary_name_fr, datasets['sentences_translation_fr_train'])
        self.target_vocab_size = self.tokenizer_fr.vocab_size + 2

        sentences_translation_both_train = tf.data.Dataset.zip(
            (datasets['sentences_translation_en_train'],
             datasets['sentences_translation_fr_train']))
        sentences_translation_both_validation = tf.data.Dataset.zip(
            (datasets['sentences_translation_en_validation'],
             datasets['sentences_translation_fr_validation']))

        tf_encode = create_tf_encoder_bilingual(
            self.tokenizer_en, self.tokenizer_fr)

        train_preprocessed = (
            sentences_translation_both_train.map(tf_encode).cache())

        val_preprocessed = (
            sentences_translation_both_validation.map(tf_encode))

        self.training_dataset = (
            train_preprocessed
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE))
        self.valid_dataset = (
            val_preprocessed
            .padded_batch(batch_size, padded_shapes=([None], [None])))
