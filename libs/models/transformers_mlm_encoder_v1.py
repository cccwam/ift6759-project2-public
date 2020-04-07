import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM, TFBertModel, TFBertMainLayer, TFXLMRobertaForMaskedLM, \
    TFPreTrainedModel, TFBertPreTrainedModel, TFBertEmbeddings, TFT5ForConditionalGeneration, T5Config
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_tf_bert import TFBertEncoder, TFBertMLMHead, BERT_INPUTS_DOCSTRING, TFBertPooler

from libs.models import transformer

logger = tf.get_logger()

"""
    Transformer model for the language model task.
    It's an Encoder Transformer only

"""


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
    intermediate_size = model_hparams["intermediate_size"]
    dropout_rate = model_hparams["dropout_rate"]

    encoder = transformer.Encoder(
        num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size,
        seq_length, dropout_rate=dropout_rate)


    final_layer = tf.keras.layers.Dense(vocab_size)

    enc_inp = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="enc_inp")
    enc_padding_mask = tf.keras.layers.Input(
        shape=(1, 1, None), dtype=tf.float32, name="enc_padding_mask")

    enc_output = encoder(enc_inp, True, enc_padding_mask)    # (batch_size, seq_length, hidden_size)


    outputs = final_layer(enc_output)  # (batch_size, seq_length, vocab_size)

    model = tf.keras.Model([enc_inp, enc_padding_mask],
                           outputs, name=name)
    model.summary(line_length=120)
    return model
