import gc

import tensorflow as tf

logger = tf.get_logger()


def load_pretrained_layers(config: dict, my_model: tf.keras.Model):
    if "pretrained_layers" in config["model"]["hyper_params"]:
        pretrained_layers = config["model"]["hyper_params"]["pretrained_layers"]
        for pretrained_layer in pretrained_layers:
            logger.info(
                f"Load pretrained layer {pretrained_layer['layer_name']} into {pretrained_layer['target_layer_name']}")
            # https://github.com/tensorflow/tensorflow/issues/32348
            pretrained_model: tf.keras.Model \
                = tf.keras.models.load_model(pretrained_layer["model_path"], compile=False)

            for l in pretrained_layer['layer_name'].split("/"):
                # pretrained_model.summary(print_fn=logger.debug)
                pretrained_model = pretrained_model.get_layer(l)

            weights = pretrained_model.get_weights()
            my_model.get_layer(pretrained_layer['target_layer_name']).set_weights(weights)

            del pretrained_model
            gc.collect()
