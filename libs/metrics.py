import numpy as np
import sacrebleu
import tensorflow as tf

from libs.data_loaders import AbstractDataloader
from libs.losses import mlm_loss

logger = tf.get_logger()


class BleuIntervalEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, dataloader: AbstractDataloader, interval=1, nsamples=50):
        tf.keras.callbacks.Callback.__init__(self)

        self._interval = interval
        self._dataloader: AbstractDataloader = dataloader
        # noinspection PyProtectedMember
        self._nsamples = min(nsamples, self._dataloader._samples_for_valid)

        if hasattr(self._dataloader.valid_dataset, "_batch_size"):
            # noinspection PyProtectedMember
            self._batch_size = self._dataloader.valid_dataset._batch_size
        else:
            self._batch_size = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._interval == 0:
            score = []
            n_step = (self._nsamples // self._batch_size)
            for x, y_true in self._dataloader.valid_dataset.take(n_step):
                y_pred = self.model.predict(x, verbose=0)
                score += [self.bleu_graph_mode(y_true, y_pred)]
            score = np.mean(score)
            tf.summary.scalar('BLEU', data=score, step=epoch)
            logger.info("BLEU Interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))

    def bleu_graph_mode(self, y_true, y_pred):
        """
            Compute the bleu score
            Code for TF2 graph mode as TF2 doesn't like Eager mode for callbacks
        Args:
            y_true: targets, shape Batch Size * Sequence
            y_pred: predictions, shape Batch Size * Sequence * Token probs

        Returns:

        """

        def compute_bleu(inputs):
            y_true_tf, y_pred_tf = inputs[0], inputs[1]

            pred_sentence = self._dataloader.decode(y_pred_tf.numpy().tolist())
            true_sentence = self._dataloader.decode(y_true_tf.numpy().tolist())
            bleu_score = sacrebleu.corpus_bleu(pred_sentence, true_sentence).score

            # To display some examples in logs
            if np.random.randint(low=0, high=self._nsamples * self._batch_size * 2) == 0:
                logger.info(f" BLEU Score {bleu_score} for {pred_sentence} - {true_sentence}")

            return tf.convert_to_tensor(bleu_score)

        stack = tf.stack([y_true, y_pred.argmax(-1)], axis=1)
        return tf.reduce_mean(tf.map_fn(compute_bleu, stack, dtype=tf.float32))


def perplexity(y_true, y_pred):
    """
    The perplexity metric.
    https://stackoverflow.com/questions/44697318/how-to-implement-perplexity-in-keras
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    cross_entropy = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.keras.backend.exp(cross_entropy)


def perplexity_mlm(y_true, y_pred):
    """
    The perplexity metric for mask language model
    """
    cross_entropy = mlm_loss(y_true, y_pred)
    return tf.keras.backend.exp(cross_entropy)
