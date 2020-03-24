import os

import tensorflow_datasets as tfds


def subword_tokenizer(vocabulary_file, sentences, target_vocab_size=2**13,
                      force_compute=False):
    """Subword tokenizer for given sentences

    :param vocabulary_file: str, path to vocabulary file on disk
    :param sentences: TextLineDataset of the sentences
    :param target_vocab_size: int
    :param force_compute: force computation even if vocabulary_file exists
    :return: tokenizer
    """

    if force_compute or (not os.path.isfile(vocabulary_file)):
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (sentence.numpy() for sentence in sentences),
            target_vocab_size=target_vocab_size)
        tokenizer.save_to_file(vocabulary_file)
    else:
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            vocabulary_file)
    return tokenizer
