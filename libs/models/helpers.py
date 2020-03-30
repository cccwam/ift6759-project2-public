import logging

import tensorflow as tf

from libs.helpers import get_online_model, load_dict

logger = logging.getLogger(__name__)


def load_pretrained_layers(config: dict, my_model: tf.keras.Model):
    if "pretrained_layers" in config["model"]["hyper_params"]:
        pretrained_layers = config["model"]["hyper_params"]["pretrained_layers"]
        for pretrained_layer in pretrained_layers:
            logger.info(
                f"Load pretrained layer {pretrained_layer['layer_name']} into {pretrained_layer['target_layer_name']}")
            try:
                pretrained_model: tf.keras.Model \
                    = tf.keras.models.load_model(pretrained_layer["model_path"])
            except (OSError, ValueError):
                print(pretrained_layer)
                pretrained_config = load_dict(pretrained_layer["model_config_path"])
                # ToDo this does not work cause the model is not built...
                pretrained_model = get_online_model(pretrained_config)
                # pretrained_model.load_weights(pretrained_layer["model_path"])
                pretrained_model.load_checkpoint()

            for l in pretrained_layer['layer_name'].split("/"):
                pretrained_model.summary(print_fn=logger.debug)
                pretrained_model = pretrained_model.get_layer(l)

            weights = pretrained_model.get_weights()
            my_model.get_layer(pretrained_layer['target_layer_name']).set_weights(weights)
