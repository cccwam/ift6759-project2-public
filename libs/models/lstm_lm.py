import logging
import typing

import tensorflow as tf

logger = logging.getLogger(__name__)


# noinspection DuplicatedCode
def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]
):
    """
    builder

    :param config: The configuration dictionary. It must follow configs/user/schema.json
    :return: tf.keras.Model
    """
    # noinspection PyShadowingNames
    def my_lstm_encoder(name: str,
                        seq_length: int,
                        vocab_size: int,
                        embedding_dim: int,
                        latent_dim: int):
        inputs = tf.keras.layers.Input(shape=(seq_length,), name=name + "_input")
        embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True,
                                               name=name + "_embedding")
        lstm_layer = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=False, name=name + "_lstm")
        outputs = lstm_layer(embeddings(inputs))

        return tf.keras.Model(inputs, outputs, name=name)

    # noinspection PyShadowingNames
    def my_head(name: str,
                seq_length: int,
                vocab_size: int,
                latent_dim: int):
        latent_inputs = tf.keras.Input(shape=(seq_length, latent_dim), name='latent')

        encoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax', name=name + "_dense")
        encoder_time_distributed = tf.keras.layers.TimeDistributed(encoder_dense, name=name + "_time_distributed")

        outputs = encoder_time_distributed(latent_inputs)

        return tf.keras.Model(latent_inputs, outputs, name=name + '_head')

    def my_model(encoder: tf.keras.Model, head: tf.keras.Model):
        # noinspection PyProtectedMember
        inputs = tf.keras.Input(shape=encoder.layers[0]._batch_input_shape[1:], name='tokens')

        x = encoder(inputs)
        x = head(x)

        return tf.keras.Model(inputs, x, name='lstm_lm')

    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size = dl_hparams["vocab_size"]
    seq_length = dl_hparams["seq_length"]
    name = model_hparams["name"]
    embedding_dim = model_hparams["embedding_dim"]
    latent_dim = model_hparams["latent_dim"]

    my_lstm_encoder = my_lstm_encoder(
        name=name,
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim)

    my_lstm_encoder.summary(print_fn=logger.info)

    my_head = my_head(name=name,
                      seq_length=seq_length,
                      vocab_size=vocab_size,
                      latent_dim=latent_dim)

    my_head.summary(print_fn=logger.info)

    my_model = my_model(my_lstm_encoder, my_head)
    my_model.summary(print_fn=logger.info)

    return my_model
