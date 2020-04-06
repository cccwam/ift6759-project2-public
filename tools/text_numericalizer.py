import logging
import pickle
from pathlib import Path

import fire

logger = tf.get_logger()
logging.basicConfig(level=logging.INFO)


def my_numericalizer(data_path):
    """
        Inspired by the notebook EDA-FM-13032020.ipynb
        This function will transform list of list of tokens into list of list of ids and save the vocabulary.

        To execute, run the following cli:   python tools/text_numericalizer.py ~/output_data

    Args:
        data_path: Data path for inputs file (filename as provided by text_normalizer.py) and output files

    Returns:

    """
    data_path = Path(data_path)

    logger.info("Load all tokenized corpora")

    with open(data_path / "unaligned_en_tokenized.pickle", 'rb') as f:
        unaligned_en_tokenized = pickle.load(f)

    with open(data_path / "train_lang1_en_tokenized.pickle", 'rb') as f:
        train_lang1_en_tokenized = pickle.load(f)

    with open(data_path / "unaligned_fr_tokenized.pickle", 'rb') as f:
        unaligned_fr_tokenized = pickle.load(f)

    with open(data_path / "train_lang2_fr_tokenized.pickle", 'rb') as f:
        train_lang2_fr_tokenized = pickle.load(f)

    """
    Preprocessing of inputs: Converts list of words into set of words
    The output is a dictionary sorted by reverse frequency (except for special tokens which are at the beginning)
    """

    def create_vocab(corpora):
        special_tokens = {"<MASK>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>>": 3}
        token_to_word = {}
        word_to_token = {}
        token_to_frequency = {}
        token_id = 0

        for data in corpora:
            for i, list_of_words in enumerate(data):
                for w in list_of_words:
                    if w not in word_to_token:
                        token_to_word[token_id] = w
                        word_to_token[w] = token_id
                        token_to_frequency[token_id] = 1
                        token_id += 1
                    else:
                        token_to_frequency[word_to_token[w]] += 1

        return {**special_tokens, **{token_to_word[k]: k + len(special_tokens) for k, _ in
                                     sorted(token_to_frequency.items(), key=lambda item: item[1], reverse=True)}}

    # EN

    logger.info("Create English vocabulary")

    word_to_token_en = create_vocab([unaligned_en_tokenized, train_lang1_en_tokenized])

    token_to_word_en = {v: k for k, v in word_to_token_en.items()}

    logger.info("Save English vocabulary")

    with open(data_path / 'word_to_token_en.pickle', 'wb') as handle:
        pickle.dump(word_to_token_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path / 'token_to_word_en.pickle', 'wb') as handle:
        pickle.dump(token_to_word_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Numericalize English corpora")

    unaligned_en_numericalized = [[word_to_token_en[w] for w in s] for s in unaligned_en_tokenized]

    train_lang1_en_numericalized = [[word_to_token_en[w] for w in s] for s in train_lang1_en_tokenized]

    logger.info("Save English numericalized corpora")

    with open(data_path / 'train_lang1_en_numericalized.pickle', 'wb') as handle:
        pickle.dump(train_lang1_en_numericalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path / 'unaligned_en_numericalized.pickle', 'wb') as handle:
        pickle.dump(unaligned_en_numericalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # FR

    logger.info("Create French vocabulary")

    word_to_token_fr = create_vocab([unaligned_fr_tokenized, train_lang2_fr_tokenized])

    token_to_word_fr = {v: k for k, v in word_to_token_fr.items()}

    logger.info("Save French vocabulary")

    with open(data_path / 'word_to_token_fr.pickle', 'wb') as handle:
        pickle.dump(word_to_token_fr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path / 'token_to_word_fr.pickle', 'wb') as handle:
        pickle.dump(token_to_word_fr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Numericalize French corpora")

    unaligned_fr_numericalized = [[word_to_token_fr[w] for w in s] for s in unaligned_fr_tokenized]

    train_lang2_fr_numericalized = [[word_to_token_fr[w] for w in s] for s in train_lang2_fr_tokenized]

    logger.info("Save French numericalized corpora")

    with open(data_path / 'train_lang2_fr_numericalized.pickle', 'wb') as handle:
        pickle.dump(train_lang2_fr_numericalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path / 'unaligned_fr_numericalized.pickle', 'wb') as handle:
        pickle.dump(unaligned_fr_numericalized, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(my_numericalizer)
