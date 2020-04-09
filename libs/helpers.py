import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import jsonschema
import tensorflow as tf

from libs.data_loaders import AbstractDataloader
from libs.losses import mlm_loss
from libs.metrics import perplexity, BleuIntervalEvaluation, perplexity_mlm
from libs.models import transformer
from libs.optimizers import CustomSchedule

logger = tf.get_logger()


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


def get_online_data_loader(config, raw_english_test_set_file_path=None):
    """
    Get an online version of the data loader defined in config

    :param config: config: The configuration dictionary. It must follow configs/user/schema.json
    :param raw_english_test_set_file_path: The raw English test set file path (Optional). For example:
        /project/cq-training-1/project2/data/test.lang1 (we don't have access to this specific file, the TAs have it)
        If this path is not specified, our data loaders will instead load the pre-processed data defined in config
    :return: An instance of config['model']['definition']['module'].['name']
    """

    return import_from(
        config['data_loader']['definition']['module'],
        config['data_loader']['definition']['name']
    )(
        config=config,
        raw_english_test_set_file_path=raw_english_test_set_file_path
    )


def get_online_model(config):
    """
    Get an online version of the model defined in config

    :param config: The configuration dictionary. It must follow configs/user/schema.json
    :return: An instance of config['model']['definition']['module'].['name']
    """

    return import_from(
        config['model']['definition']['module'],
        config['model']['definition']['name']
    )(
        config=config
    )


def prepare_model(config):
    """
    Prepare model

    Args:
        config: The configuration dictionary. It must follow configs/user/schema.json

    Returns:
        A ``tf.keras.Model`` object that can be used to generate French sentences given English sentence tensors.
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

    print(f"Loading model: {model_source}")
    if config["model"]["definition"]["module"] == 'libs.models.transformerv2':
        model = transformer.load_transformer(config)
    else:
        model = tf.keras.models.load_model(model_source)
    return model


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


def compile_model(model,
                  learning_rate: float,
                  dataloader: AbstractDataloader,
                  loss: str,
                  optimizer: str,
                  config: dict,
                  metrics: List[str] = None):
    """
        Helper function to compile a new model at each variation of the experiment
    :param learning_rate:
    :param dataloader: dataloader
    :param loss: loss function name
    :param optimizer: optimizer function name
    :param model: model to be compiled
    :param metrics: list of metrics
    :param config: configuration dictionary
    :return: compiled model and additional callbacks (for metrics which are too slow to run on training set)
    """

    mapping_metrics = {
        "perplexity": perplexity,  # For language model task
        "perplexity_mlm": perplexity_mlm,
        #        "bleu": bleu,  # For translation task but it's too slow to run during training
        "sparse_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),  # Generic for classification
    }

    mapping_loss = {
        "sparse_categorical_cross_entropy": tf.keras.losses.SparseCategoricalCrossentropy(),
        "mlm_loss": mlm_loss
    }

    if "d_model" in config['model']['hyper_params']:  # From Blaise model
        d_model = config['model']['hyper_params']['d_model']
    else:
        if "hidden_size" in config['model']['hyper_params']:  # From François model
            d_model = config['model']['hyper_params']['hidden_size']
        else:
            if optimizer == "adam-transformer":
                raise Exception("adam-transformer requires d_model or hidden size in config")
            d_model = 1  # Never executed but prevents a warning in PyCharm
    mapping_optimizer = {
        "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        "adam-transformer": tf.keras.optimizers.Adam(
            CustomSchedule(d_model), beta_1=0.9, beta_2=0.98,  # TODO test with diff learning rate
            epsilon=1e-9)
    }

    metric_funcs, additional_callbacks = [], []
    for metric in metrics:
        if metric == "bleu":
            # Special case
            #  To be called only on end of epoch because it's too slow
            additional_callbacks += [BleuIntervalEvaluation(dataloader=dataloader)]
        else:
            metric_funcs += [mapping_metrics[metric]]

    optimizer = mapping_optimizer[optimizer]

    model.compile(
        optimizer=optimizer,
        loss=mapping_loss[loss],
        metrics=metric_funcs
    )

    return model, additional_callbacks


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
