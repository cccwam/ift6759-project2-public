import logging

import tensorflow as tf

from libs.helpers import get_online_model, load_dict, get_online_data_loader

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
                pretrained_config = load_dict(pretrained_layer["model_config_path"])
                print("Retrieving data loader for pretraining task")
                pretrained_data_loader = get_online_data_loader(pretrained_config)
                pretrained_data_loader.build(
                    batch_size=64,
                    mode=pretrained_config['data_loader']['hyper_params']['mode'])
                training_dataset, valid_dataset = \
                    pretrained_data_loader.training_dataset, pretrained_data_loader.valid_dataset
                pretrained_model = get_online_model(pretrained_config)
                print("Initial fit on 1 batch to build pretraining model")
                pretrained_model.fit(
                    training_dataset.take(1), validation_steps=2,
                    ckpt_manager=None)
                print("Loading weights")
                pretrained_model.load_weights(pretrained_layer["model_path"])

            for l in pretrained_layer['layer_name'].split("/"):
                # pretrained_model.summary(print_fn=logger.debug)
                pretrained_model = pretrained_model.get_layer(l)

            weights = pretrained_model.get_weights()
            my_model.get_layer(pretrained_layer['target_layer_name']).set_weights(weights)
