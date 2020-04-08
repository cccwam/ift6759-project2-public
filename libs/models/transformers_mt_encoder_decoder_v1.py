import typing

import tensorflow as tf

from libs.models import transformer
from libs.models.helpers import load_pretrained_layers

logger = tf.get_logger()

"""
    Transformers model for the translation task.

"""


def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size_source = dl_hparams["vocab_size_source"]
    seq_length_source = dl_hparams["seq_length_source"]

    vocab_size_target = dl_hparams["vocab_size_target"]
    seq_length_target = dl_hparams["seq_length_target"]

    # For the config see https://github.com/google-research/bert
    name = model_hparams["name"]
    num_hidden_layers = model_hparams["num_hidden_layers"]
    num_attention_heads = model_hparams["num_attention_heads"]
    hidden_size = model_hparams["hidden_size"]
    intermediate_size = model_hparams["intermediate_size"]
    dropout_rate = model_hparams["dropout_rate"]

    encoder = transformer.Encoder(
        num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size_source,
        seq_length_source, dropout_rate=dropout_rate)

    decoder = transformer.Decoder(
        num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size_target,
        seq_length_target, dropout_rate=dropout_rate)

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

    enc_output = encoder(enc_inp, True, enc_padding_mask)  # (batch_size, seq_length, hidden_size)

    dec_output, attention_weights = decoder(
        dec_inp, enc_output, True, combined_mask, dec_padding_mask)

    outputs = final_layer(dec_output)  # (batch_size, seq_length, vocab_size)

    model = tf.keras.Model([enc_inp, dec_inp, enc_padding_mask, combined_mask, dec_padding_mask],
                           outputs, name=name)

    model.summary(line_length=120)
    load_pretrained_layers(config=config, my_model=model)
    return model
