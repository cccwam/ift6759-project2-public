import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM, TFBertModel, TFBertMainLayer, TFXLMRobertaForMaskedLM, \
    TFPreTrainedModel, TFBertPreTrainedModel, TFBertEmbeddings, TFT5ForConditionalGeneration, T5Config
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_tf_bert import TFBertEncoder, TFBertMLMHead, BERT_INPUTS_DOCSTRING, TFBertPooler

from libs.models import transformer

logger = tf.get_logger()

"""
    BERT model for the translation task and translation language model task.
    It's Encoder-Decoder Transformer only

    THIS IS THE MODEL USED FOR THE PROJECT 2 OF IFT6759
"""


def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size_source = dl_hparams["vocab_size_source"]
    seq_length_source = dl_hparams["seq_length_source"] # TODO review

    vocab_size_target = dl_hparams["vocab_size_target"]
    seq_length_target = dl_hparams["seq_length_target"]

    # For the config see https://github.com/google-research/bert
    name = model_hparams["name"]
    num_hidden_layers = model_hparams["num_hidden_layers"]
    num_attention_heads = model_hparams["num_attention_heads"]
    hidden_size = model_hparams["hidden_size"]
    dropout_rate = model_hparams["dropout_rate"]

    encoder = transformer.Encoder(
        num_hidden_layers, hidden_size, num_attention_heads, seq_length_source, vocab_size_source,
        vocab_size_source, dropout_rate=dropout_rate)

    decoder = transformer.Decoder(
        num_hidden_layers, hidden_size, num_attention_heads, seq_length_target, vocab_size_target,
        vocab_size_target, dropout_rate=dropout_rate)

    final_layer = tf.keras.layers.Dense(vocab_size_target)

    enc_inp = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="enc_inp")
    dec_inp = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="dec_inp")
    enc_padding_mask = tf.keras.layers.Input(
        shape=(1, 1, None), dtype=tf.float32, name="enc_padding_mask")
    combined_mask = tf.keras.layers.Input(
        shape=(1, None, None), dtype=tf.float32, name="combined_mask")
    dec_padding_mask = tf.keras.layers.Input(
        shape=(1, 1, None), dtype=tf.float32, name="dec_padding_mask")

    enc_output = encoder(enc_inp, True, enc_padding_mask)
    # (batch_size, inp_seq_len, hidden_size)

    # dec_output.shape == (batch_size, tar_seq_len, hidden_size)
    dec_output, attention_weights = decoder(
        dec_inp, enc_output, True, combined_mask, dec_padding_mask)

    outputs = final_layer(
        dec_output)  # (batch_size, tar_seq_len, vocab_size_target)

    model = tf.keras.Model([enc_inp, dec_inp, enc_padding_mask, combined_mask, dec_padding_mask],
                           outputs, name=name)
    model.summary(line_length=120)
    return model
