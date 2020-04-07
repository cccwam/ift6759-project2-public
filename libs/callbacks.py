import logging
from pathlib import Path

import tensorflow as tf
from transformers import TFPreTrainedModel

logger = tf.get_logger()


class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def _save_model(self, epoch, logs):
        """ Same as tf.keras.callbacks.ModelCheckpoint except that it saves also the HuggingFace model
            with the HuggingFace method save_pretrained

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)

                        # CUSTOM LOGIC
                        self._custom_logic(filepath=filepath)
                        # END CUSTOM LOGIC
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

                # CUSTOM LOGIC
                self._custom_logic(filepath=filepath)
                # END CUSTOM LOGIC

            self._maybe_remove_file()

    def _custom_logic(self, filepath: str):
        if isinstance(self.model.layers[-1], TFPreTrainedModel):
            model_hgf: TFPreTrainedModel = self.model.layers[-1]
            folder: Path = Path(filepath).parent / "huggingface"
            folder.mkdir(parents=True, exist_ok=True)
            model_hgf.save_pretrained(str(folder))



