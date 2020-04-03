# Code source: https://www.tensorflow.org/tutorials/text/transformer
import typing

import tensorflow as tf

from libs.helpers import get_online_data_loader
from libs.models import transformer
from libs.models.helpers import load_pretrained_layers


class TransformerV2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size_source,
                 vocab_size_target, pe_input, pe_target, rate=0.1,
                 model_name='TransformerV2'):
        super(TransformerV2, self).__init__()

        self.encoder = transformer.Encoder(
            num_layers, d_model, num_heads, dff, vocab_size_source, pe_input,
            rate)

        self.decoder = transformer.Decoder(
            num_layers, d_model, num_heads, dff, vocab_size_target, pe_target,
            rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size_target)

        # ToDo investigate how this overwrites the logic in trainer.py
        self.lrv2 = transformer.CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.lrv2, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def call(self, inputs, training):
        enc_inp, dec_inp, padding_mask, combined_mask = inputs
        enc_output = self.encoder(enc_inp, training, padding_mask)
        # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            dec_inp, enc_output, training, combined_mask, padding_mask)

        final_output = self.final_layer(
            dec_output)  # (batch_size, tar_seq_len, vocab_size_target)

        return final_output, attention_weights

    # ToDo Reimplement evaluate (with call & training=False?)


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

    transformer_tl = TransformerV2(
        num_layers, d_model, num_heads, dff, vocab_size_source,
        vocab_size_target, pe_input=vocab_size_source,
        pe_target=vocab_size_target, rate=dropout_rate,
        model_name=model_hparams["name"])

    if "pretrained_layers" in config["model"]["hyper_params"]:
        print("Entering pretraining procedure")
        print("Retrieving data loader")
        task_data_loader = get_online_data_loader(config)
        task_data_loader.build(
            batch_size=64, mode=config['data_loader']['hyper_params']['mode'])
        training_dataset, _ = \
            task_data_loader.training_dataset, task_data_loader.valid_dataset
        print("Initial fit on 1 batch to build main task model")
        transformer_tl.fit(
            training_dataset.take(1), validation_steps=2, ckpt_manager=None)
        print("Loading pretrained layers")
        load_pretrained_layers(config, transformer_tl)
        print("Completed loading weights from pretrained model")

    return transformer_tl
