import spacy
import fire
import logging
import numpy as np
from sklearn.utils import shuffle
from pathlib import Path
import pandas as pd
import pickle
import tqdm

logger = logging.getLogger(__name__)


def my_tokenizer(data_path,
                 output_path,
                 should_shuffle=True,
                 shuffle_seed=42):
    """
        Inspired by the notebook EDA-FM-13032020.ipynb
        This function will normalize monolingual and bilingual corpora.

        To execute, run the following cli:   python tools/text_normalizer.py ~/data/ ~/output_data

    Args:
        data_path: Data path for inputs file (filename as provided by TAs)
        output_path: Output path for the pickle files (containing list of list of tokens)
        should_shuffle: Boolean to indicate if should shuffle
        shuffle_seed: Seed

    Returns:

    """
    logger.basicConfig(level=logger.INFO)

    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initialize Spacy tokenizers")

    # python -m spacy download en_core_web_sm
    tokenizer_en = spacy.load("en_core_web_sm")

    # Add back the sentencizer because the tokenizer will run without other parts
    tokenizer_en.add_pipe(tokenizer_en.create_pipe('sentencizer'))

    # python -m spacy download fr_core_news_sm
    tokenizer_fr = spacy.load("fr_core_news_sm")

    # Add back the sentencizer because the tokenizer will run without other parts
    tokenizer_fr.add_pipe(tokenizer_fr.create_pipe('sentencizer'))

    logger.info("Read all inputs files")

    with open(data_path / "unaligned.en", 'r') as f:
        unaligned_en = [line.rstrip() for line in f]  # Remove the \n
        unaligned_en = pd.DataFrame(unaligned_en, columns=["text"])
        f.close()

    with open(data_path / "train.lang1", 'r') as f:
        train_lang1_en = [line.rstrip() for line in f]  # Remove the \n
        train_lang1_en = pd.DataFrame(train_lang1_en, columns=["text"])
        f.close()

    with open(data_path / "unaligned.fr", 'r') as f:
        unaligned_fr = [line.rstrip() for line in f]  # Remove the \n
        unaligned_fr = pd.DataFrame(unaligned_fr, columns=["text"])
        f.close()

    with open(data_path / "train.lang2", 'r') as f:
        train_lang2_fr = [line.rstrip() for line in f]  # Remove the \n
        train_lang2_fr = pd.DataFrame(train_lang2_fr, columns=["text"])
        f.close()

    # Inspired by TA's function
    # TODO move to helper ? It must be used for the evaluator
    # noinspection PyUnboundLocalVariable
    def my_tokenize(input_dataframe, tokenizer, keep_case, keep_punctuation, split_sentence=False):
        for i, row in tqdm.tqdm(input_dataframe.text.iteritems(), total=input_dataframe.shape[0]):
            # Remove NER, dependency parser and POS to speed up
            doc = tokenizer(row, disable=["parser", "entity", "ner"])
            if split_sentence:
                sentence = []
            for sent in doc.sents:
                tokens_list = [w for w in sent]
                if not keep_punctuation:
                    tokens_list = [t for t in tokens_list if
                                   not t.is_punct or t.is_quote or t.text in ["-", "%", "--", "–", "/", "..", "…", "&",
                                                                              "“", "”"]]
                tokens_list = [t.text for t in tokens_list]
                if not keep_case:
                    tokens_list = [w.lower() for w in tokens_list]
                if len(tokens_list) == 0:
                    continue
                if split_sentence:
                    sentence += tokens_list
                else:
                    yield tokens_list
            if split_sentence:
                yield sentence

    # EN
    logger.info("Tokenize English corpora")

    unaligned_en_tokenized = list(my_tokenize(unaligned_en, tokenizer=tokenizer_en,
                                              keep_case=False, keep_punctuation=False))
    # Required also to extend the vocab
    train_lang1_en_tokenized = list(my_tokenize(train_lang1_en, tokenizer=tokenizer_en,
                                                keep_case=False, keep_punctuation=False))

    # FR
    logger.info("Tokenize French corpora")

    unaligned_fr_tokenized = list(my_tokenize(unaligned_fr, tokenizer=tokenizer_fr,
                                              keep_case=True, keep_punctuation=True))
    # Required also to extend the vocab
    train_lang2_fr_tokenized = list(my_tokenize(train_lang2_fr, tokenizer=tokenizer_fr,
                                                keep_case=True, keep_punctuation=True, split_sentence=True))

    assert len(train_lang2_fr_tokenized) == len(
        train_lang1_en_tokenized), "The bilingual dataset must match in number of samples"

    if should_shuffle:
        logger.info("Shuffle corpora")
        if shuffle_seed:
            np.random.seed(shuffle_seed)
        train_lang1_en_tokenized, train_lang2_fr_tokenized = shuffle(train_lang1_en_tokenized, train_lang2_fr_tokenized)
        unaligned_en_tokenized = shuffle(unaligned_en_tokenized)
        unaligned_fr_tokenized = shuffle(unaligned_fr_tokenized)

    logger.info("Save all corpora")

    with open(output_path / 'unaligned_en_tokenized.pickle', 'wb') as handle:
        pickle.dump(unaligned_en_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'train_lang1_en_tokenized.pickle', 'wb') as handle:
        pickle.dump(train_lang1_en_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'unaligned_fr_tokenized.pickle', 'wb') as handle:
        pickle.dump(unaligned_fr_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'train_lang2_fr_tokenized.pickle', 'wb') as handle:
        pickle.dump(train_lang2_fr_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(my_tokenizer)
