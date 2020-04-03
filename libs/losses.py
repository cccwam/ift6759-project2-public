import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# TODO double check because HGF has got 4/5 for loss where I've got 0.25
# https://huggingface.co/blog/how-to-train
def mlm_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
