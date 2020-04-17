import sacrebleu
import tensorflow as tf
from tensorflow_probability.python.distributions import Binomial

from libs.data_loaders import AbstractDataloader
from libs.losses import mlm_loss

logger = tf.get_logger()


class BleuIntervalEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, dataloader: AbstractDataloader, interval=1, nsamples=50, n_samples_to_show=2):
        tf.keras.callbacks.Callback.__init__(self)

        self._interval = interval
        self._dataloader: AbstractDataloader = dataloader

        if hasattr(self._dataloader, "_samples_for_valid"):
            self._nsamples = min(nsamples, self._dataloader.samples_for_valid)

        self._n_samples_to_show = n_samples_to_show

        if hasattr(self._dataloader, "batch_size"):
            self._batch_size = self._dataloader.batch_size
        else:
            self._batch_size = 1

        # No need to seed here. This is only used to activate logging event not affecting metric computations
        self._show_sentences_dist = Binomial(total_count=1, probs=(self._n_samples_to_show / self._batch_size))

    def on_epoch_end(self, epoch, logs=None):
        # https://github.com/keras-team/keras/issues/13671
        @tf.function(experimental_relax_shapes=True)
        def mypred(inputs):
            return self.model(inputs)

        if epoch % self._interval == 0:
            score = 0
            n_step = (self._nsamples // self._batch_size)
            for x, y_true in self._dataloader.valid_dataset.take(n_step):
                y_pred = mypred(x)
                score += self.bleu_graph_mode(y_true, y_pred)

            score /= n_step
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
            show_sentence = self._show_sentences_dist.sample(sample_shape=[1])
            show_sentence = tf.cast(show_sentence, dtype=tf.int32)
            if tf.equal(show_sentence[0], 1):
                logger.info(f" BLEU Score {bleu_score} for {pred_sentence} - {true_sentence}")

            return bleu_score

        stack = tf.stack([y_true, tf.argmax(y_pred, -1, output_type=tf.int32)], axis=1)
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
