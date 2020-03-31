# Code source: https://www.tensorflow.org/tutorials/text/transformer
import os
import time
import typing

import tensorflow as tf

from libs.models import transformer
from libs.helpers import loss_function_for_transformer as loss_function


class TransformerLeftLM(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerLeftLM, self).__init__()

        self.encoder = transformer.Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input,
            rate)

        self.left_lm_final_layer = tf.keras.layers.Dense(input_vocab_size)

        self.lr = transformer.CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='validation_accuracy')

    def call(self, tar, training, enc_padding_mask, look_ahead_mask,
             dec_padding_mask):

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        lm_output = self.encoder(
            tar, training, look_ahead_mask)

        final_output = self.left_lm_final_layer(
            lm_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def load_checkpoint(self):
        # ToDo customize checkpoint save directory
        checkpoint_path = os.path.join(os.environ['HOME'],
                                       "ift6759_p2_checkpoints/left_lm")
        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                  max_to_keep=3)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        return ckpt_manager

    def fit(self, x=None, epochs=1, callbacks=None, validation_data=None,
            validation_steps=None, **kwargs):
        ckpt_manager = kwargs['ckpt_manager']

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = transformer.create_masks(
                inp, tar_inp)

            with tf.GradientTape() as tape:
                # ToDo for the left language model, we need to mask the
                # first input, double check this?
                predictions = self.call(tar_inp, True, combined_mask,
                                        combined_mask, dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)

        @tf.function(input_signature=train_step_signature)
        def validation_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = transformer.create_masks(
                inp, tar_inp)

            # ToDo make sure mask is ok
            predictions = self.call(tar_inp, True, combined_mask,
                                    combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)

            self.validation_loss(loss)
            self.validation_accuracy(tar_real, predictions)

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp0, tar0)) in enumerate(x):
                train_step(inp0, tar0)

                if batch % 50 == 0:
                    print(
                        'Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                            epoch + 1, batch, self.train_loss.result(),
                            self.train_accuracy.result()))

            if (epoch + 1) % 2 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(
                    'Saving checkpoint for epoch {} at {}'.format(
                        epoch + 1, ckpt_save_path))

            print(
                'Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, self.train_loss.result(),
                    self.train_accuracy.result()))

            if (epoch + 1) % validation_steps == 0:
                self.validation_loss.reset_states()
                self.validation_accuracy.reset_states()

                for (batch, (inp0, tar0)) in enumerate(validation_data):
                    validation_step(inp0, tar0)

                print(
                    'Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}'.format(
                        epoch + 1, self.validation_loss.result(),
                        self.validation_accuracy.result()))

            print(
                'Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def builder(config: typing.Dict[typing.AnyStr, typing.Any]):
    # noinspection PyShadowingNames,DuplicatedCode
    model_hparams = config["model"]["hyper_params"]

    num_layers = model_hparams["num_layers"]
    d_model = model_hparams["d_model"]
    num_heads = model_hparams["num_heads"]
    dff = model_hparams["dff"]
    dropout_rate = model_hparams["dropout_rate"]
    input_vocab_size = model_hparams["input_vocab_size"]
    target_vocab_size = model_hparams["target_vocab_size"]

    return TransformerLeftLM(
        num_layers, d_model, num_heads, dff, input_vocab_size,
        target_vocab_size, pe_input=input_vocab_size,
        pe_target=target_vocab_size, rate=dropout_rate)
