# ift6759-project2

Project 2 solution from Team 3 in IFT6759: Advanced Projects
at University of Montreal.

Team members:
* Blaise Gauvin St-Denis 
* François Mercier 
* Helgi Tomas Gislason
* Ilyas Amaniss

## Notable files

* trainer.py
  * Used to train the predictor.
* evaluator.py
  * Used to evaluate the predictor.
* tools/
  * Helper scripts to facilitate development.
* requirements.txt
  * All the requirements necessary to run the code.

## Guide to our configuration files

In this project, we utilise a user configuration file schema can be found at configs/user/schema.json. All in all, it
contains the following configurable properties:

* model
    * definition:
        * module: The Python module that contains the definition for the model that should be used
        * name: The Python name that is the model definition
    * hyper_params: The hyper parameters for the model (some are defined as a list to allow
    for a hyper-parameter search during training)
    * source: This parameter is used when the code is run using evaluator.py. It determines
    how to load the trained model. It's value should be one of the following:
        * "" (defaults loading model during evaluation to ../model/best_model.hdf5)
        * Absolute path to the hdf5 file 
        * "online": Means that the model does not have a .hdf5 file and it will be loaded
        without any predefined weights. 
* data_loader
    * definition:
        * module: The Python module that contains the definition for the data_loader that should be used
        * name: The Python name that is the data_loader definition
    * hyper_params: The hyper parameters for the data_loader
    * should_preprocess_data: Whether or not to pre-process the data before training/validating/testing.
    * preprocessed_data_source:
        * training: Location of the training data.
        * validation: Location of the validation data.
        * test: Location of the testing data.
* trainer 
    * hyper_params: We have defined that the following trainer hyper parameters are required by our trainer:
        * lr_rate: The learning rate
        * batch_size: The batch size
        * epochs: The number of epochs
        * patience: How patient we should be before we perform early stopping during training

## Sbatch job example

The the folder tools, there's a file called sbatch_template.sh. 
It is currently set up to run a training of our best model configuration. 
Simply run `sbatch sbatch_template.sh` to launch the training job.

## Evaluator.py example usage

```
python evaluator.py \
    --input-file-path predictions.txt \
    --target-file-path targets.txt \
    --do-not-run-model
```

* `input-file-path`: path to input file
* `target-file-path`: path to target (reference) file
* `print-all-scores`: will print one score per sentence
* `do-not-run-model`: (Default: True) will use --input-file-path as predictions, instead of running the model on it

## Trainer.py example usage

Note: Read the section `Sbatch job example` above for information on how to run an sbatch job to train a model.

```
python trainer.py \
    --?? To be defined
    --user_cfg_path configs/user/sample.json \
    --tensorboard_tracking_folder /project/cq-training-1/project2/teams/team03/tensorboard/$USER
```

* `??` To be defined
* `user_cfg_path`: Path to the JSON config file used to store user model, dataloader and trainer parameters.
* `tensorboard_tracking_folder`: Path where to store TensorBoard data and save trained model. 
