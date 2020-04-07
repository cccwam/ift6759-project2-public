import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM

from libs.models.bert_mlm_v1 import builder_mlm

logger = tf.get_logger()

"""
    BERT model for the translation task and translation language model task.
    It's an Encoder Transformer only

    THIS IS THE MODEL USED FOR THE PROJECT 2 OF IFT6759
"""


def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]

    vocab_size_source = dl_hparams["vocab_size_source"]
    seq_length_source = dl_hparams["seq_length_source"]

    vocab_size_target = dl_hparams["vocab_size_target"]
    seq_length_target = dl_hparams["seq_length_target"]

    assert vocab_size_source == vocab_size_target, "vocab_size_source and vocab_size_target must be the same for BERT"

    consolidated_seq_length = seq_length_source + seq_length_target

    config["data_loader"]["hyper_params"]["vocab_size"] = vocab_size_target
    config["data_loader"]["hyper_params"]["seq_length"] = consolidated_seq_length

    return builder_mlm(config=config)
