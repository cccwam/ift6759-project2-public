import logging
import pickle
import tempfile
from pathlib import Path

# ig
import fire
from tokenizers.implementations import BaseTokenizer

from libs.helpers import import_from

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_bpe_tokenizer(data_path,
                        output_path,
                        tokenizer_algorithm="ByteLevelBPETokenizer",
                        vocab_size_fr=20000,
                        vocab_size_en=20000,
                        dropout=0.1,  # https://jlibovicky.github.io/2019/11/07/MT-Weekly-BPE-dropout.html
                        tmp_path="/tmp"):
    """
        Script to train specifically one tokenizer model without training a DL model.
        WARNING: Please use the config file and the associated dataloader before trying this solution.

    Args:
        data_path: Data path for inputs file (filename as provided by text_normalizer.py) and output files
        output_path: Output_path for the saved tokenizer model
        tokenizer_algorithm: HuggingFaces implementation (ByteLevelBPETokenizer, CharBPETokenizer,
        SentencePieceBPETokenizer or BertWordPieceTokenizer)
        vocab_size_fr: vocabulary size for the French (target)
        vocab_size_en: vocabulary size for the English (source)
        dropout: Dropout hyperparameter for the tokenizer
        tmp_path: Optional. HuggingFaces requires to save corpus into file. A temp file is used for this.

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

    logger.info(f"Train {tokenizer_algorithm}")

    tokenizer_fr: BaseTokenizer = import_from("tokenizers", tokenizer_algorithm)(dropout=dropout)

    with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
        raw_consolidated_corpus = [" ".join(l) for l in train_lang2_fr_tokenized + unaligned_fr_tokenized]
        tmp.writelines(raw_consolidated_corpus)
        tokenizer_fr.train(files=[tmp.name], show_progress=True, vocab_size=vocab_size_fr)

    tokenizer_en: BaseTokenizer = import_from("tokenizers", tokenizer_algorithm)(dropout=dropout)

    with tempfile.NamedTemporaryFile(mode="w", dir=tmp_path) as tmp:
        raw_consolidated_corpus = [" ".join(l) for l in train_lang1_en_tokenized + unaligned_en_tokenized]
        tmp.writelines(raw_consolidated_corpus)
        tokenizer_en.train(files=[tmp.name], show_progress=True, vocab_size=vocab_size_en)

    logger.info("Compute the max length for each language")

    tokenized_texts_fr = tokenizer_fr.encode_batch(
        [" ".join(l) for l in train_lang2_fr_tokenized + unaligned_fr_tokenized])
    seq_length_fr = max([len(tokenized_text_fr.ids) for tokenized_text_fr in tokenized_texts_fr])
    tokenized_texts_en = tokenizer_en.encode_batch(
        [" ".join(l) for l in train_lang1_en_tokenized + unaligned_en_tokenized])
    seq_length_en = max([len(tokenized_text_en.ids) for tokenized_text_en in tokenized_texts_en])
    logger.info(f"Max length for FR: {seq_length_fr}, Max length for EN: {seq_length_en}")

    logger.info("Save BPE tokenizer")

    tokenizer_fr.save(output_path, f"FR_{tokenizer_algorithm}_vocab_size_{vocab_size_fr}_dropout_{dropout}")
    tokenizer_en.save(output_path, f"EN_{tokenizer_algorithm}_vocab_size_{vocab_size_fr}_dropout_{dropout}")


if __name__ == '__main__':
    fire.Fire(train_bpe_tokenizer)
