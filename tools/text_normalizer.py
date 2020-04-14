import logging
import pickle
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import spacy
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def my_tokenizer(data_path,
                 output_path,
                 should_shuffle=True,
                 shuffle_seed=42):
    """
        Inspired by the notebook EDA-FM-13032020.ipynb
        This function will normalize monolingual and bilingual corpora.

        To execute, run the following cli:   python tools/text_normalizer.py ~/data/ ~/output_data

        Ex:  'python tools/text_normalizer.py /project/cq-training-1/project2/data/
                /project/cq-training-1/project2/teams/team03/data/preprocessed_13042020'

    Args:
        data_path: Data path for inputs file (filename as provided by TAs)
        output_path: Output path for the pickle files (containing list of list of tokens)
                    and text files (same but text format)
        should_shuffle: Boolean to indicate if should shuffle
        shuffle_seed: Seed

    Returns:

    """
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
    # noinspection PyUnboundLocalVariable
    def my_tokenize(input_dataframe, tokenizer, keep_case, keep_punctuation, split_sentence=False):
        for i, row in tqdm.tqdm(input_dataframe.text.iteritems(), total=input_dataframe.shape[0]):
            # Remove NER, dependency parser and POS to speed up
            doc = tokenizer(row, disable=["parser", "entity", "ner"])
            if not split_sentence:
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
                if not split_sentence:
                    sentence += tokens_list
                else:
                    yield tokens_list
            if not split_sentence:
                yield sentence

    # EN
    logger.info("Tokenize English corpora")

    # unaligned_en = unaligned_en[:2]  # Only for debug
    # train_lang1_en = train_lang1_en[:2]
    # unaligned_fr = unaligned_fr[:2]
    # train_lang2_fr = train_lang2_fr[:2]

    unaligned_en_tokenized = np.array(list(my_tokenize(unaligned_en, tokenizer=tokenizer_en,
                                                       keep_case=False, keep_punctuation=False)))

    unaligned_en = np.array([" ".join(list_tokens) for list_tokens in unaligned_en_tokenized])

    # Required also to extend the vocab
    train_lang1_en_tokenized = np.array(list(my_tokenize(train_lang1_en, tokenizer=tokenizer_en,
                                                         keep_case=False, keep_punctuation=False)))

    # FR
    logger.info("Tokenize French corpora")

    unaligned_fr_tokenized = np.array(list(my_tokenize(unaligned_fr, tokenizer=tokenizer_fr,
                                                       keep_case=True, keep_punctuation=True)))

    unaligned_fr = np.array([" ".join(list_tokens) for list_tokens in unaligned_fr_tokenized])

    # Required also to extend the vocab
    train_lang2_fr_tokenized = np.array(np.array(list(my_tokenize(train_lang2_fr, tokenizer=tokenizer_fr,
                                                                  keep_case=True, keep_punctuation=True))))

    assert len(train_lang2_fr_tokenized) == len(
        train_lang1_en_tokenized), "The bilingual dataset must match in number of samples"

    train_lang1_en = train_lang1_en.text.to_numpy()
    train_lang2_fr = train_lang2_fr.text.to_numpy()

    assert len(train_lang1_en_tokenized) == len(train_lang1_en)
    assert len(train_lang2_fr_tokenized) == len(train_lang2_fr)

    if should_shuffle:
        logger.info("Shuffle corpora")
        if shuffle_seed:
            np.random.seed(shuffle_seed)

        idx = np.arange(len(train_lang2_fr_tokenized))
        train_lang1_en = train_lang1_en[idx]
        train_lang2_fr = train_lang2_fr[idx]
        train_lang1_en_tokenized = train_lang1_en_tokenized[idx]
        train_lang2_fr_tokenized = train_lang2_fr_tokenized[idx]

        idx = np.arange(len(unaligned_en))
        unaligned_en = unaligned_en[idx]
        unaligned_en_tokenized = unaligned_en_tokenized[idx]

        idx = np.arange(len(unaligned_en))
        unaligned_fr = unaligned_fr[idx]
        unaligned_fr_tokenized = unaligned_fr_tokenized[idx]

    logger.info("Save all corpora")

    with open(output_path / 'unaligned_en_tokenized.pickle', 'wb') as handle:
        pickle.dump(unaligned_en_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'unaligned_en_tokenized.txt', 'w') as handle:
        for s in unaligned_en:
            handle.write(s + "\n")

    with open(output_path / 'train_lang1_en_tokenized.pickle', 'wb') as handle:
        pickle.dump(train_lang1_en_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'train_lang1_en.txt', 'w') as handle:
        for s in train_lang1_en:
            handle.write(s + "\n")

    with open(output_path / 'unaligned_fr_tokenized.pickle', 'wb') as handle:
        pickle.dump(unaligned_fr_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'unaligned_fr_tokenized.txt', 'w') as handle:
        for s in unaligned_fr:
            handle.write(s + "\n")

    with open(output_path / 'train_lang2_fr_tokenized.pickle', 'wb') as handle:
        pickle.dump(train_lang2_fr_tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / 'train_lang2_fr.txt', 'w') as handle:
        for s in train_lang2_fr:
            handle.write(s + "\n")


if __name__ == '__main__':
    fire.Fire(my_tokenizer)
