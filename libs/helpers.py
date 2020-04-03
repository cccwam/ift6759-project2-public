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


def get_model(
        config
):
    """
    Get model

    Returns the model as defined in the config. The config should follow configs/user/schema.json

    Args:
        config: The configuration dictionary. It must follow configs/user/schema.json

    Returns:
        A ``tf.keras.Model`` object that can be used to generate French sentences given English sentence tensors.
    """
    mirrored_strategy = get_mirrored_strategy()

    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            model = prepare_model(config)
    else:
        model = prepare_model(config)

    return model


def prepare_model(
        config
):
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

    # ToDo Better handling of models that support .load_model vs those who don't
    try:
        model = tf.keras.models.load_model(model_source)
    except ValueError:
        print("Retrieving data loader for task")
        data_loader = get_online_data_loader(config)
        data_loader.build(
            batch_size=64,
            mode=config['data_loader']['hyper_params']['mode'])
        training_dataset, _ = \
            data_loader.training_dataset, data_loader.valid_dataset
        model = get_online_model(config)
        print("Initial fit on 1 batch to build model")
        model.fit(
            training_dataset.take(1), validation_steps=2,
            ckpt_manager=None)
        print("Loading weights")
        model.load_weights(model_source)

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


def compile_model(model, learning_rate, d_model=None):
    """
        Helper function to compile a new model at each variation of the experiment
    :param learning_rate:
    :param model:
    :return:
    """

    model_instance = model

    # ToDo identify how to select optimizer
    if learning_rate == -1:
        optimizer = tf.keras.optimizers.Adam(
            CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # TODO to review this
    #   Likely that we should have more than 1 loss
    #   Also we should be able to optimizer
    # Add also BLEU
    if learning_rate == -1:
        model_instance.compile(
            optimizer=optimizer,
            loss=loss_function_for_transformer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    else:
        model_instance.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
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


def loss_function_for_transformer(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model_cast = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        # config = super().get_config().copy()
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model_cast) * tf.math.minimum(arg1, arg2)
