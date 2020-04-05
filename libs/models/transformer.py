"""Transformer neural network

Notes:
    Most code blocks are taken from
    https://www.tensorflow.org/tutorials/text/transformer

"""

import os
import time
import typing

import tensorflow as tf
import numpy as np

# ToDo Remove this when obsolete
# from libs.helpers import get_online_data_loader
# from libs.models.helpers import load_pretrained_layers


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # shape of attention_weights is now (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
        })
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        # shape of q is now (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        # shape of k is now (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)
        # shape of v is now (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape ==
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # shape of scaled_attention is (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        # shape of concat_attention is (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)
        # shape of output is (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        # above layer has shape (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        # atten_output has shape (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # out1 has shape (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        # out2 has shape (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # attn1 has shape (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        # attn2 has shape (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        # out2 has shape (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        # out3 has shape (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size_source,
                 maximum_position_encoding, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size_source = vocab_size_source
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(vocab_size_source, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'vocab_size_source': self.vocab_size_source,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size_target,
                 maximum_position_encoding, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size_target = vocab_size_target
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size_target, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'vocab_size_target': self.vocab_size_target,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask,
                                                   padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# ToDo obsolete?
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size_source,
                 vocab_size_target, pe_input, pe_target, dropout_rate=0.1,
                 model_name='Transformer'):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               vocab_size_source, pe_input, dropout_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               vocab_size_target, pe_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size_target)

        self.lr = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='validation_accuracy')
        self.model_name = model_name

    def call(self, enc_inp, dec_inp, training, enc_padding_mask, look_ahead_mask,
             dec_padding_mask):
        enc_output = self.encoder(enc_inp, training, enc_padding_mask)
        # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            dec_inp, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(
            dec_output)  # (batch_size, tar_seq_len, vocab_size_target)

        return final_output, attention_weights

    def load_checkpoint(self):
        checkpoint_path = os.path.join(
            os.environ['HOME'], f"ift6759_p2_checkpoints/{self.model_name}")
        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                  max_to_keep=3)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        return ckpt_manager

    def fit(self, x=None, epochs=1, callbacks=None,
            validation_data=None, validation_steps=None, **kwargs):
        ckpt_manager = kwargs['ckpt_manager']

        @tf.function(experimental_relax_shapes=True)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = self(inp, tar_inp, True, enc_padding_mask,
                                      combined_mask, dec_padding_mask)
                loss = loss_function_for_transformer(tar_real, predictions)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)

        @tf.function(experimental_relax_shapes=True)
        def validation_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp, tar_inp)

            predictions, _ = self.call(inp, tar_inp, True, enc_padding_mask,
                                       combined_mask, dec_padding_mask)
            loss = loss_function_for_transformer(tar_real, predictions)

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

            if (epoch + 1) % 5 == 0:
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

    def evaluate(self, encoder_input, transformer_output, end_token):
        # This limit in i should be MAX_LENGTH, but no longer using it so...
        for slen in range(100):
            enc_padding_mask, combined_mask, dec_padding_mask = \
                create_masks(encoder_input, transformer_output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self(encoder_input,
                                                  transformer_output,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                return tf.squeeze(transformer_output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the
            # decoder as its input.
            transformer_output = tf.concat([transformer_output, predicted_id], axis=-1)

        return tf.squeeze(transformer_output, axis=0), attention_weights


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# ToDo obsolete?
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

    transformer_tl = Transformer(
        num_layers, d_model, num_heads, dff, vocab_size_source,
        vocab_size_target, pe_input=vocab_size_source,
        pe_target=vocab_size_target, dropout_rate=dropout_rate,
        model_name=model_hparams["name"])

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

    return transformer_tl


def loss_function_for_transformer(real, pred):
    """Loss function for transformer model.

    :param real: target
    :param pred: prediction
    :return: loss

    """

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule for transformer."""

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model_cast = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        # config = super().get_config().copy()
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model_cast) * tf.math.minimum(arg1, arg2)


def load_transformer(config):
    """Load transformer model

    :param config: dictionary, parameters of model, including source file
    :return: tf.keras.Model, the restored transformer model from file

    """

    return tf.keras.models.load_model(
        config['model']['source'],
        custom_objects={'Encoder': Encoder,
                        'Decoder': Decoder,
                        'CustomSchedule': CustomSchedule,
                        'loss_function_for_transformer': loss_function_for_transformer})


def inference(tokenizer, model, test_dataset):
    """Inference step for transformer.

    :param tokenizer: tokenizer used for sentence reconstruction
    :param model: tf.keras.Model, the model to use
    :param test_dataset: tokenized test sentences to translate
    :return: list of sentences, the translated sentences

    """

    begin_token = tokenizer.vocab_size
    end_token = tokenizer.vocab_size + 1

    # ToDo actually fetch by name
    encoder = model.layers[3]
    # encoder = [layer for layer in model.layers if layer.name=='Encoder'][0]
    decoder = model.layers[5]
    # decoder = [layer for layer in model.layers if layer.name == 'Decoder'][0]
    final_layer = model.layers[6]
    # final_layer = [layer for layer in model.layers if layer.name == '???'][0]
    all_predictions = []
    for i, test_inp in enumerate(test_dataset):
        # ToDo better verbosity
        print(i)
        enc_inp, dec_inp, padding_mask, combined_mask = test_inp
        enc_output = encoder(enc_inp, False, padding_mask)
        # ToDo allow different max length?
        for slen in range(100):
            dec_output, attention_weights = decoder(
                dec_inp, enc_output, False, combined_mask,
                padding_mask)
            final_output = final_layer(dec_output)
            # final_output.shape == (batch_size, seq_len, vocab_size)

            # select the last word from the seq_len dimension
            predictions = final_output[:, -1:, :]
            # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1),
                                   tf.int32)  # (batch_size, 1)

            # ToDo reimplement this stop criteria in batch?
            # return the result if the predicted_id is equal to the end token
            # if predicted_id == end_token:
            #     return tf.squeeze(transformer_output,
            #                       axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the
            # decoder as its input.
            dec_inp = tf.concat([dec_inp, predicted_id], axis=-1)

        for i in range(dec_inp.shape[0]):
            sent_ids = []
            for j in dec_inp[i]:
                if j == begin_token:
                    continue
                if j == end_token:
                    break
                sent_ids.append(j)
            predicted_sentence = tokenizer.decode(sent_ids)
            all_predictions.append(predicted_sentence)

    return all_predictions
