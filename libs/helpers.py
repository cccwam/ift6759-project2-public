import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import jsonschema
import tensorflow as tf

logger = logging.getLogger(__name__)


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def load_dict(json_file_path):
    with open(json_file_path, 'r') as file_handler:
        file_data = file_handler.read()

    return json.loads(file_data)


def validate_user_config(config):
    schema_file = load_dict('configs/user/schema.json')
    jsonschema.validate(config, schema_file)


def get_online_data_loader(config):
    """
    # TODO review doc
    Get an online version of the data loader defined in config

    If admin_config_dict is not specified, the following have to be specified:
        * dataframe
        * target_datetimes
        * stations
        * target_time_offsets
    If admin_config_dict is specified, it overwrites the parameters specified above.

    :param config: The user dictionary used to store user model/dataloader parameters
    :return: An instance of config['model']['definition']['module'].['name']
    """

    return import_from(
        config['data_loader']['definition']['module'],
        config['data_loader']['definition']['name']
    )(
        config=config
    )


def get_online_model(config):
    """
    # TODO review doc
    Get an online version of the model defined in config

    If admin_config_dict is not specified, the following have to be specified:
        * stations
        * target_time_offsets
    If admin_config_dict is specified, it overwrites the parameters specified above.

    :param config: The user dictionary used to store user model/dataloader parameters
    :return: An instance of config['model']['definition']['module'].['name']
    """

    return import_from(
        config['model']['definition']['module'],
        config['model']['definition']['name']
    )(
        config=config
    )


def prepare_model(
        config
):
    """
    Prepare model
    # TODO review doc

    Args:
        config: configuration dictionary holding any extra parameters that might be required by the user.
            These parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
    """
    default_model_path = '../model/best_model.hdf5'
    model_source = config['model']['source']

    if model_source == 'online':
        return get_online_model(config=config)
    elif model_source:
        if not os.path.exists(model_source):
            raise FileNotFoundError(f'Error: The file {model_source} does not exist.')
    else:
        if os.path.exists(default_model_path):
            model_source = default_model_path
        else:
            raise FileNotFoundError(f'Error: The file {default_model_path} does not exist.')

    return tf.keras.models.load_model(model_source)


def generate_model_name(config):
    return "{}.{}.{}.hdf5".format(
        config['model']['definition']['module'],
        config['model']['definition']['name'],
        uuid.uuid4().hex
    )


def get_tensorboard_experiment_id(experiment_name, tensorboard_tracking_folder: Path):
    """
    Create a unique id for TensorBoard for the experiment

    :param experiment_name: name of experiment
    :param tensorboard_tracking_folder: Path where to store TensorBoard data and save trained model
    """
    model_sub_folder = experiment_name + "-" + datetime.utcnow().isoformat()
    return tensorboard_tracking_folder / model_sub_folder


def compile_model(model, learning_rate):
    """
        Helper function to compile a new model at each variation of the experiment
    :param learning_rate:
    :param model:
    :return:
    """

    model_instance = model

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # TODO to review this
    #   Likely that we should have more than 1 loss
    #   Also we should be able to optimizer
    model_instance.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model_instance


def get_module_name(module_dictionary):
    return module_dictionary["definition"]["module"].split(".")[-1] + "." + module_dictionary["definition"]["name"]


def get_mirrored_strategy():
    nb_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(2, nb_gpus))])
    logger.debug("------------")
    logger.debug('Number of available GPU devices: {}'.format(nb_gpus))
    logger.debug('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    logger.debug("------------")
    return mirrored_strategy
