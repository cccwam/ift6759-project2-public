import tensorflow as tf

logger = tf.get_logger()


@tf.function
def mlm_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # Batch size * Seq Length

    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # Prevent averaging over non masked tokens
    numerator = tf.reduce_sum(loss_, axis=1)
    denominator = tf.reduce_sum(mask, axis=1)
    # For numerical stability
    denominator = tf.clip_by_value(denominator, clip_value_min=1.,
                                   clip_value_max=tf.cast(tf.reduce_sum(tf.shape(y_true)), dtype=denominator.dtype))
    loss = numerator / denominator

    # For Debug only
    # if tf.math.is_nan(tf.reduce_mean(loss)):
    #     logger.error("Nan value in loss")
    #     tf.print(y_pred)

    return tf.reduce_mean(loss)  # Average over samples
