import logging
import typing

import tensorflow as tf

from libs.models.helpers import load_pretrained_layers

logger = logging.getLogger(__name__)


# noinspection DuplicatedCode
def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    # noinspection PyShadowingNames,DuplicatedCode
    def my_seq2seq(name: str,
                   seq_length_source: int,
                   vocab_size_source: int,
                   seq_length_target: int,
                   vocab_size_target: int,
                   embedding_dim: int,
                   latent_dim: int):
        encoder_name = name + "_encoder"
        encoder_inputs = tf.keras.layers.Input(shape=(seq_length_source,), name=encoder_name + "_input")
        embeddings = tf.keras.layers.Embedding(vocab_size_source, embedding_dim, mask_zero=True,
                                               name=encoder_name + "_embedding")
        decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True,
                                            name=encoder_name + "_lstm")
        _, state_h, state_c = decoder_lstm(embeddings(encoder_inputs))
        encoder_states = [state_h, state_c]

        decoder_name = name + "_decoder"
        decoder_inputs = tf.keras.layers.Input(shape=(seq_length_target,), name=decoder_name + "_input")
        embeddings = tf.keras.layers.Embedding(vocab_size_target, embedding_dim, mask_zero=True,
                                               name=decoder_name + "_embedding")
        decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True,
                                            name=decoder_name + "_lstm")
        decoder_outputs, _, _ = decoder_lstm(embeddings(decoder_inputs),
                                             initial_state=encoder_states)

        decoder_dense = tf.keras.layers.Dense(vocab_size_target, activation='softmax', name=decoder_name + "_dense")
        decoder_time_distributed = tf.keras.layers.TimeDistributed(decoder_dense,
                                                                   name=decoder_name + "_time_distributed")
        decoder_outputs = decoder_time_distributed(decoder_outputs)

        return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name=name)


    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size_source = dl_hparams["vocab_size_source"]
    seq_length_source = dl_hparams["seq_length_source"]

    vocab_size_target = dl_hparams["vocab_size_target"]
    seq_length_target = dl_hparams["seq_length_target"]

    name = model_hparams["name"]
    embedding_dim = model_hparams["embedding_dim"]
    latent_dim = model_hparams["latent_dim"]

    my_model = my_seq2seq(
        name=name,
        seq_length_source=seq_length_source,
        vocab_size_source=vocab_size_source,
        seq_length_target=seq_length_target,
        vocab_size_target=vocab_size_target,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim)

    my_model.summary(print_fn=logger.info)

    load_pretrained_layers(config=config, my_model=my_model)

    return my_model
