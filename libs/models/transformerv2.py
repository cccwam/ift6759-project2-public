# Code source: https://www.tensorflow.org/tutorials/text/transformer
import typing

import tensorflow as tf

from libs.models import transformer


def builder(config: typing.Dict[typing.AnyStr, typing.Any]):
    # noinspection PyShadowingNames,DuplicatedCode
    model_hparams = config["model"]["hyper_params"]
    dl_hparams = config["data_loader"]["hyper_params"]

    num_layers = model_hparams["num_layers"]
    d_model = model_hparams["d_model"]
    num_heads = model_hparams["num_heads"]
    dff = model_hparams["dff"]
    dropout_rate = model_hparams["dropout_rate"]
    vocab_size_source = dl_hparams["vocab_size_source"]
    vocab_size_target = dl_hparams["vocab_size_target"]

    encoder = transformer.Encoder(
        num_layers, d_model, num_heads, dff, vocab_size_source,
        vocab_size_source, dropout_rate=dropout_rate)

    decoder = transformer.Decoder(
        num_layers, d_model, num_heads, dff, vocab_size_target,
        vocab_size_target, dropout_rate=dropout_rate)

    final_layer = tf.keras.layers.Dense(vocab_size_target)

    enc_inp = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int64, name="i1")
    dec_inp = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int64, name="i2")
    padding_mask = tf.keras.layers.Input(
        shape=(1, 1, None), dtype=tf.float32, name="i3")
    combined_mask = tf.keras.layers.Input(
        shape=(1, None, None), dtype=tf.float32, name="i4")

    enc_output = encoder(enc_inp, True, padding_mask)
    # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = decoder(
        dec_inp, enc_output, True, combined_mask, padding_mask)

    outputs = final_layer(
        dec_output)  # (batch_size, tar_seq_len, vocab_size_target)

    model = tf.keras.Model([enc_inp, dec_inp, padding_mask, combined_mask],
                           outputs, name=model_hparams["name"])
    model.summary(line_length=120)

    # ToDo review this, probably obsolete
    # if "pretrained_layers" in config["model"]["hyper_params"]:
    #     print("Entering pretraining procedure")
    #     print("Retrieving data loader")
    #     task_data_loader = get_online_data_loader(config)
    #     task_data_loader.build(
    #         batch_size=64, mode=config['data_loader']['hyper_params']['mode'])
    #     training_dataset, _ = \
    #         task_data_loader.training_dataset, task_data_loader.valid_dataset
    #     print("Initial fit on 1 batch to build main task model")
    #     transformer_tl.fit(
    #         training_dataset.take(1), validation_steps=2, ckpt_manager=None)
    #     print("Loading pretrained layers")
    #     load_pretrained_layers(config, transformer_tl)
    #     print("Completed loading weights from pretrained model")

    return model
