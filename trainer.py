# Summary:
#   Trains the predictor

import argparse
import logging
import os
import typing
from pathlib import Path
from typing import List

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from libs import helpers
from libs.data_loaders.abstract_dataloader import AbstractDataloader

import numpy as np
import random
from libs.seeds import TENSOR_FLOW_SEED, NUMPY_SEED, RANDOM_SEED

tf.random.set_seed(TENSOR_FLOW_SEED)
np.random.seed(NUMPY_SEED)
random.seed(RANDOM_SEED)

logger = tf.get_logger()


def main(
        config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    """
    Train a model

    :param config_path: path to the JSON config file that follows configs/user/schema.json
    :param tensorboard_tracking_folder: path where to store TensorBoard data and save trained model
    """
    user_config_dict = helpers.load_dict(config_path)
    helpers.validate_user_config(user_config_dict)

    if tensorboard_tracking_folder is not None:
        tensorboard_tracking_folder = Path(tensorboard_tracking_folder)
        tensorboard_tracking_folder.mkdir(parents=True, exist_ok=True)

    train_models(
        config=user_config_dict,
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )


def train_models(
        config: dict,
        tensorboard_tracking_folder: Path
):
    """
    Train the possible combinations of models based on the hyper parameters defined in  config

    :param config: The configuration dictionary. It must follow configs/user/schema.json
    :param tensorboard_tracking_folder: The TensorBoard tracking folder
    """
    model_dict = config['model']
    model_dict_hparams = config['model']['hyper_params']
    data_loader_dict = config['data_loader']
    trainer_hyper_params = config['trainer']['hyper_params']

    data_loader: AbstractDataloader = helpers.get_online_data_loader(config)

    mirrored_strategy = helpers.get_mirrored_strategy()

    logger.info(f"\nModel definitions: {model_dict}\n")
    logger.info(f"\nData loader definitions: {data_loader_dict}\n")
    logger.info(f"\nTrainer hyper parameters: {trainer_hyper_params}\n")

    model_name = helpers.get_module_name(model_dict)
    data_loader_name = helpers.get_module_name(data_loader_dict)

    hp_model = hp.HParam('model_class', hp.Discrete([model_name]))
    hp_model_hparams = [hp.HParam(k, hp.Discrete([v])) for k, v in model_dict_hparams.items()
                        if k not in ["pretrained_layers"]]
    if "pretrained_layers" in model_dict_hparams:
        pretrained_layers = []
        for pretrained_layer in model_dict_hparams["pretrained_layers"]:
            pretrained_layer_id = f"{pretrained_layer['model_path']}"
            pretrained_layer_id = pretrained_layer_id + "-" + f"{pretrained_layer['layer_name']}"
            pretrained_layer_id = pretrained_layer_id + "-" + f"{pretrained_layer['target_layer_name']}"
            pretrained_layers += [pretrained_layer_id]
        hp_model_hparams += [hp.HParam("pretrained_layers", hp.Discrete(["-".join(pretrained_layers)]))]
    hp_model_hparams = {hparam: hparam.domain.values[0] for hparam in hp_model_hparams}

    hp_dataloader = hp.HParam('dataloader_class', hp.Discrete([data_loader_name]))
    hp_dataloader_hparams = hp.HParam('dataloader_id', hp.Discrete([data_loader.get_hparams()]))

    tensorboard_experiment_name = model_name + "_" + data_loader_name
    if tensorboard_tracking_folder is not None:
        tensorboard_experiment_id = helpers.get_tensorboard_experiment_id(
            experiment_name=tensorboard_experiment_name,
            tensorboard_tracking_folder=tensorboard_tracking_folder
        )
    else:
        tensorboard_experiment_id = tensorboard_experiment_name

    # Hyper parameters search for the training loop
    hp_batch_size = hp.HParam('batch_size', hp.Discrete(trainer_hyper_params["batch_size"]))
    hp_epochs = hp.HParam('epochs', hp.Discrete(trainer_hyper_params["epochs"]))
    if "dropout" in model_dict['hyper_params'].keys():
        hp_dropout = hp.HParam('dropout', hp.Discrete(model_dict['hyper_params']["dropout"]))
    else:
        # Value to indicate no dropout for the model.
        hp_dropout = hp.HParam('dropout', hp.Discrete([-1.0]))
    hp_learning_rate = hp.HParam('learning_rate', hp.Discrete(trainer_hyper_params["lr_rate"]))
    hp_patience = hp.HParam('patience', hp.Discrete(trainer_hyper_params["patience"]))

    data_loader.build(batch_size=hp_batch_size.domain.values[0])

    # Main loop to iterate over all possible hyper parameters
    variation_num = 0
    for epochs in hp_epochs.domain.values:
        for learning_rate in hp_learning_rate.domain.values:
            for dropout in hp_dropout.domain.values:
                for patience in hp_patience.domain.values:
                    hparams = {
                        hp_batch_size: hp_batch_size.domain.values[0],
                        hp_model: hp_model.domain.values[0],
                        hp_dataloader: hp_dataloader.domain.values[0],
                        hp_dataloader_hparams: hp_dataloader_hparams.domain.values[0],
                        hp_epochs: epochs,
                        hp_learning_rate: learning_rate,
                        hp_patience: patience,
                    }

                    if dropout != -1.:
                        hparams[hp_dropout] = dropout

                    # model hparams to tensorboard
                    hparams = {**hparams, **hp_model_hparams}

                    # Copy the user config for the specific current model
                    current_user_dict = config.copy()
                    # Add dropout
                    if dropout != -1.:
                        current_user_dict["model"]['hyper_params']["dropout"] = dropout

                    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
                        with mirrored_strategy.scope():
                            model = helpers.prepare_model(config)
                    else:
                        model = helpers.prepare_model(config)

                    if tensorboard_tracking_folder is not None:
                        tensorboard_log_dir = str(tensorboard_experiment_id / str(variation_num))
                        # TF format is needed to load layers and not just model (hdf5 doesn't work)
                        checkpoints_path = str(tensorboard_log_dir) + "/" + (tensorboard_experiment_name +
                                                                             ".{epoch:02d}-{val_loss:.2f}.tf")
                        logger.info(f"Start variation id: " + str(tensorboard_log_dir))
                    else:
                        tensorboard_log_dir, checkpoints_path = None, None
                        logger.info(f"Start variation id: {tensorboard_experiment_id} {str(variation_num)}")

                    train_model(
                        model=model,
                        dataloader=data_loader,
                        tensorboard_log_dir=tensorboard_log_dir,
                        hparams=hparams,
                        mirrored_strategy=mirrored_strategy,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        loss=trainer_hyper_params["loss"],
                        optimizer=trainer_hyper_params["optimizer"],
                        metrics=trainer_hyper_params["metrics"],
                        patience=patience,
                        checkpoints_path=checkpoints_path,
                        config=config
                    )
                    variation_num += 1

    # Save final model
    if tensorboard_tracking_folder is not None:
        model.save(os.path.join(tensorboard_experiment_id, '0', 'final_model.hdf5'))
    else:
        model_name = helpers.generate_model_name(config)
        model.save(model_name)


def train_model(
        model: tf.keras.Model,
        dataloader: AbstractDataloader,
        tensorboard_log_dir: str,
        hparams,
        mirrored_strategy,
        epochs: int,
        learning_rate: float,
        loss: str,
        optimizer: str,
        metrics: List[str],
        checkpoints_path: str,
        config: dict,
        patience: int,
):
    """
    The training loop for a single model
    :param model: The tf.keras.Model to train
    :param dataloader: The dataloader
    :param tensorboard_log_dir: Path of where to store TensorFlow logs
    :param hparams: A dictionary of TensorBoard.plugins.hparams.api.hp.HParam to track on TensorBoard
    :param mirrored_strategy: A tf.distribute.MirroredStrategy on how many GPUs to use during training
    :param epochs: The epochs hyper parameter
    :param learning_rate: The learning rate hyper parameter
    :param loss: loss function name
    :param optimizer: optimizer function name
    :param patience: The early stopping patience hyper parameter
    :param checkpoints_path: Path of where to store TensorFlow checkpoints
    :param metrics: list of metrics
    :param config: configuration dictionary
    """

    if tensorboard_log_dir is not None:

        # For the BLEU score which is computed in a callback on epoch end
        file_writer = tf.summary.create_file_writer(tensorboard_log_dir + "/validation")
        file_writer.set_as_default()

        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, profile_batch=0),
                     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_weights_only=False,
                                                        save_best_only=False, monitor='val_loss'),
                     hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams)]
    else:
        callbacks = []

    # Multi GPU setup
    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            compiled_model, additional_callbacks = helpers.compile_model(model=model,
                                                                         dataloader=dataloader,
                                                                         loss=loss,
                                                                         optimizer=optimizer,
                                                                         metrics=metrics,
                                                                         learning_rate=learning_rate,
                                                                         config=config)
    else:
        compiled_model, additional_callbacks = helpers.compile_model(model=model,
                                                                     dataloader=dataloader,
                                                                     loss=loss,
                                                                     optimizer=optimizer,
                                                                     metrics=metrics,
                                                                     learning_rate=learning_rate,
                                                                     config=config)

    if patience != -1:
        callbacks += [tf.keras.callbacks.EarlyStopping(patience=patience)] + additional_callbacks
    else:
        callbacks += additional_callbacks

    compiled_model.fit(
        dataloader.training_dataset,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=dataloader.train_steps if hasattr(dataloader, "train_steps") else None,
        validation_data=dataloader.valid_dataset,
        validation_steps=dataloader.validation_steps
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='path to the JSON config that follows configs/user/schema.json')
    parser.add_argument('-t', '--tensorboard_tracking_folder', type=str, default=None,
                        help='path where to store TensorBoard data and save trained model')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Start")

    main(
        config_path=args.config,
        tensorboard_tracking_folder=args.tensorboard_tracking_folder
    )

    logger.info("End")
