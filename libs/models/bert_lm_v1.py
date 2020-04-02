import logging
import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM

logger = logging.getLogger(__name__)


def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size = dl_hparams["vocab_size"]
    seq_length = dl_hparams["seq_length"]

    # For the config see https://github.com/google-research/bert
    name = model_hparams["name"]
    num_hidden_layers = model_hparams["num_hidden_layers"]
    num_attention_heads = model_hparams["num_attention_heads"]
    hidden_size = model_hparams["hidden_size"]

    configuration = BertConfig(vocab_size=vocab_size,
                               num_hidden_layers=num_hidden_layers,
                               num_attention_heads=num_attention_heads,
                               max_position_embeddings=seq_length,
                               hidden_size=hidden_size)

    # Initializing a model from the configuration
    bert_model = TFBertForMaskedLM(configuration)

    token_inputs = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32, name="BERT_token_inputs")
    outputs = bert_model(token_inputs)

    model = tf.keras.Model(token_inputs, outputs, name=name)

    model.summary(line_length=120, print_fn=logger.info)

    return model
