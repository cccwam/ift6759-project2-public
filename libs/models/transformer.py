"""Transformer neural network

Notes:
    Most code blocks are taken from
    https://www.tensorflow.org/tutorials/text/transformer

"""

from typing import List

import numpy as np
import tensorflow as tf

from libs import losses


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

        self.wq: tf.keras.layers.Dense = tf.keras.layers.Dense(d_model)
        self.wk: tf.keras.layers.Dense = tf.keras.layers.Dense(d_model)
        self.wv: tf.keras.layers.Dense = tf.keras.layers.Dense(d_model)

        self.dense: tf.keras.layers.Dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, v=None, k=None, q=None, mask=None, training=None):
        assert inputs is None, "You should not use inputs"  # Needed to match Keras signature
        assert v is not None, "You should use v"
        assert k is not None, "You should use k"
        assert q is not None, "You should use q"
        assert mask is not None, "You should use mask"

        batch_size = tf.shape(q)[0]

        q = self.wq(inputs=q)  # (batch_size, seq_len, d_model)
        k = self.wk(inputs=k)  # (batch_size, seq_len, d_model)
        v = self.wv(inputs=v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(x=q, batch_size=batch_size)
        # shape of q is now (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(x=k, batch_size=batch_size)
        # shape of k is now (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(x=v, batch_size=batch_size)
        # shape of v is now (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape ==
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q=q, k=k, v=v, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # shape of scaled_attention is (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        # shape of concat_attention is (batch_size, seq_len_q, d_model)

        output = self.dense(inputs=concat_attention, training=training)
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

        self.mha: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
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

    def call(self, inputs, mask=None, training=None):
        attn_output, _ = self.mha(inputs=None, v=inputs, k=inputs, q=inputs, mask=mask, training=training)
        # atten_output has shape (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(inputs=attn_output, training=training)
        out1 = self.layernorm1(inputs=inputs + attn_output, training=training)
        # out1 has shape (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(inputs=ffn_output, training=training)
        out2 = self.layernorm2(inputs=out1 + ffn_output, training=training)
        # out2 has shape (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha1: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.mha2: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1: tf.keras.layers.LayerNormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2: tf.keras.layers.LayerNormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3: tf.keras.layers.LayerNormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1: tf.keras.layers.Dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2: tf.keras.layers.Dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3: tf.keras.layers.Dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def call(self, inputs, enc_output=None, look_ahead_mask=None, padding_mask=None, training=None):
        assert enc_output is not None, "You should have enc_output"
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(inputs=None,
                                               v=inputs, k=inputs, q=inputs, mask=look_ahead_mask, training=training)
        # attn1 has shape (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(inputs=attn1, training=training)
        out1 = self.layernorm1(inputs=attn1 + inputs, training=training)

        attn2, attn_weights_block2 = self.mha2(inputs=None,
                                               v=enc_output, k=enc_output, q=out1, mask=padding_mask,
                                               training=training)
        # attn2 has shape (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(inputs=attn2, training=training)
        out2 = self.layernorm2(inputs=attn2 + out1, training=training)
        # out2 has shape (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(inputs=ffn_output, training=training)
        out3 = self.layernorm3(inputs=ffn_output + out2, training=training)
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
                 maximum_position_encoding, dropout_rate=0.1):
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

        self.enc_layers: List[EncoderLayer] = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
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

    def call(self, inputs, mask=None, training=None):
        seq_len = tf.shape(inputs)[1]

        # adding embedding and position encoding.
        inputs = self.embedding(inputs=inputs, training=training)  # (batch_size, input_seq_len, d_model)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :seq_len, :]

        inputs = self.dropout(inputs=inputs, training=training)

        for i in range(self.num_layers):
            inputs = self.enc_layers[i](inputs=inputs, mask=mask, training=training)

        return inputs  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size_target,
                 maximum_position_encoding, dropout_rate=0.1):
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

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size_target, output_dim=d_model)
        self.pos_encoding = positional_encoding(position=maximum_position_encoding,
                                                d_model=d_model)

        self.dec_layers: List[DecoderLayer] = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

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

    def call(self, inputs, enc_output=None, look_ahead_mask=None, padding_mask=None, training=None):
        assert enc_output is not None, "You should have enc_output"
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        inputs = self.embedding(inputs=inputs, training=training)  # (batch_size, target_seq_len, d_model)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :seq_len, :]

        inputs = self.dropout(inputs=inputs, training=training)

        for i in range(self.num_layers):
            inputs, block1, block2 = self.dec_layers[i](inputs=inputs, enc_output=enc_output,
                                                        look_ahead_mask=look_ahead_mask,
                                                        padding_mask=padding_mask, training=training)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # inputs.shape == (batch_size, target_seq_len, d_model)
        return inputs, attention_weights


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
                        'mlm_loss': losses.mlm_loss})


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
        enc_output = encoder(inputs=enc_inp, mask=padding_mask, training=False)
        # ToDo allow different max length?
        for slen in range(100):
            dec_output, attention_weights = decoder(
                inputs=dec_inp, enc_output=enc_output, look_ahead_mask=combined_mask,
                padding_mask=padding_mask, training=False)
            final_output = final_layer(inpts=dec_output)
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

        for timestep in range(dec_inp.shape[0]):
            sent_ids = []
            for j in dec_inp[timestep]:
                if j == begin_token:
                    continue
                if j == end_token:
                    break
                sent_ids.append(j)
            predicted_sentence = tokenizer.decode(sent_ids)
            all_predictions.append(predicted_sentence)

    return all_predictions
