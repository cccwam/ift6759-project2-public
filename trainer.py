# Summary:
#   Trains the predictor

import argparse
import logging
import typing
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from libs import helpers
from libs.data_loaders.abstract_dataloader import AbstractDataloader

logger = logging.getLogger(__name__)


def main(
        config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    """
    Train a model

    :param config_path: path to the JSON config file used to store user model, dataloader and trainer parameters
    :param tensorboard_tracking_folder: path where to store TensorBoard data and save trained model
    """
    user_config_dict = helpers.load_dict(config_path)
    helpers.validate_user_config(user_config_dict)

    if tensorboard_tracking_folder is not None:
        Path(tensorboard_tracking_folder).mkdir(parents=True, exist_ok=True)

    train_models(
        config=user_config_dict,
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )


def train_models(
        config: dict,
        tensorboard_tracking_folder: Path
):
    """
    # TODO review doc
    Train the possible combinations of models based on the hyper parameters defined in  config

    :param config: A dictionary of the user configuration json file which contains the hyper parameters
    :param tensorboard_tracking_folder: The TensorBoard tracking folder
    """
    model_dict = config['model']
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
    hp_dataloader = hp.HParam('dataloader_class', hp.Discrete([data_loader_name]))

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
    training_dataset, valid_dataset = data_loader.training_dataset, data_loader.valid_dataset

    # Main loop to iterate over all possible hyper parameters
    variation_num = 0
    for epochs in hp_epochs.domain.values:
        for learning_rate in hp_learning_rate.domain.values:
            for dropout in hp_dropout.domain.values:
                for patience in hp_patience.domain.values:
                    if dropout != -1.:
                        hparams = {
                            hp_batch_size: hp_batch_size.domain.values[0],
                            hp_model: hp_model.domain.values[0],
                            hp_dataloader: hp_dataloader.domain.values[0],
                            hp_epochs: epochs,
                            hp_dropout: dropout,
                            hp_learning_rate: learning_rate,
                            hp_patience: patience,
                        }
                    else:
                        hparams = {
                            hp_batch_size: hp_batch_size.domain.values[0],
                            hp_model: hp_model.domain.values[0],
                            hp_dataloader: hp_dataloader.domain.values[0],
                            hp_epochs: epochs,
                            hp_learning_rate: learning_rate,
                            hp_patience: patience,
                        }

                    # Copy the user config for the specific current model
                    current_user_dict = config.copy()
                    # Add dropout
                    if dropout != -1.:
                        current_user_dict["model"]['hyper_params']["dropout"] = dropout

                    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
                        with mirrored_strategy.scope():
                            model = helpers.get_online_model(config)
                    else:
                        model = helpers.get_online_model(config)

                    if tensorboard_tracking_folder is not None:
                        tensorboard_log_dir = tensorboard_experiment_id / str(variation_num)
                        # Fileformat must be hdf5, otherwise bug
                        # https://github.com/tensorflow/tensorflow/issues/34127
                        checkpoints_path = tensorboard_log_dir / (tensorboard_experiment_name +
                                                                  ".{epoch:02d}-{val_loss:.2f}.hdf5")
                        logger.info(f"Start variation id: " + str(tensorboard_log_dir))
                    else:
                        tensorboard_log_dir, checkpoints_path = None, None
                        logger.info(f"Start variation id: f{tensorboard_experiment_id} {str(variation_num)}")

                    train_model(
                        model=model,
                        training_dataset=training_dataset,
                        valid_dataset=valid_dataset,
                        validation_steps=data_loader.validation_steps,
                        tensorboard_log_dir=tensorboard_log_dir,
                        hparams=hparams,
                        mirrored_strategy=mirrored_strategy,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        patience=patience,
                        checkpoints_path=checkpoints_path
                    )
                    variation_num += 1

    # Save final model
    model.save(helpers.generate_model_name(config))


def train_model(
        model: tf.keras.Model,
        training_dataset: tf.data.Dataset,
        valid_dataset: tf.data.Dataset,
        validation_steps: int,
        tensorboard_log_dir,
        hparams,
        mirrored_strategy,
        epochs,
        learning_rate,
        patience,
        checkpoints_path
):
    """
    The training loop for a single model

    :param model: The tf.keras.Model to train
    :param training_dataset: The training dataset
    :param valid_dataset: The validation dataset to evaluate training progress
    :param validation_steps: Number of minibatch for the validation dataset
    :param tensorboard_log_dir: Path of where to store TensorFlow logs
    :param hparams: A dictionary of TensorBoard.plugins.hparams.api.hp.HParam to track on TensorBoard
    :param mirrored_strategy: A tf.distribute.MirroredStrategy on how many GPUs to use during training
    :param epochs: The epochs hyper parameter
    :param learning_rate: The learning rate hyper parameter
    :param patience: The early stopping patience hyper parameter
    :param checkpoints_path: Path of where to store TensorFlow checkpoints

    """

    # Multi GPU setup
    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            compiled_model = helpers.compile_model(model, learning_rate=learning_rate)
    else:
        compiled_model = helpers.compile_model(model, learning_rate=learning_rate)

    if tensorboard_log_dir is not None:
        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), profile_batch=0),
                     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_weights_only=False,
                                                        monitor='val_loss'),
                     hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams)]
    else:
        callbacks = []

    callbacks += [
        tf.keras.callbacks.EarlyStopping(patience=patience)
    ]

    compiled_model.fit(
        training_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_dataset,
        validation_steps=validation_steps
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='path to the JSON config file used to store user model, dataloader and trainer parameters')
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
