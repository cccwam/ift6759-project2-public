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
  * Repository requirements. Run `pip install -r requirements.txt` to install.

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
        * preprocessed_data: Properties of pre-processed data to use during training
        * samples_for_train: Number of pre-processed samples for training
        * samples_for_valid: Number of pre-processed samples for validation
        * samples_for_test: Number of pre-processed samples for testing
* trainer 
    * hyper_params: We have defined that the following trainer hyper parameters are required by our trainer:
        * loss: The name of the loss function to use
        * optimizer: The name of the optimizer to use
        * lr_rate: The learning rate
        * metrics: Which metrics to monitor during training
        * batch_size: The batch size
        * epochs: The number of epochs
        * patience: How patient we should be before we perform early stopping during training
            

## Sbatch job example

In the folder tools, there's a file called sbatch_template.sh. 
It is currently set up to run a training of our best model configuration. 
Simply run `sbatch sbatch_template.sh` to launch the training job.

## Evaluator.py example usage

* `input-file-path`: path to input file (training data or pre-calculated predictions)
* `target-file-path`: path to target (reference) file
* `print-all-scores`: will print one score per sentence
* `do-not-run-model`: If specified, `--input-file-path` will be treated as predictions. Otherwise, as testing data

**Examples:**

*Produce predictions and then calculate BLEU score:*

```
python evaluator.py \
    --input-file-path /project/cq-training-1/project2/teams/team03/data/split_01/original/train.lang1.test \
    --target-file-path /project/cq-training-1/project2/teams/team03/data/split_01/original/train.lang2.test
```

*Read predictions from file and then calculate BLEU score:*

```
python evaluator.py \
    --input-file-path predictions.txt \
    --target-file-path /project/cq-training-1/project2/teams/team03/data/split_01/original/train.lang2.test \
    --do-not-run-model
```

## Trainer.py example usage

Note: Read the section `Sbatch job example` above for information on how to run an sbatch job to train a model.

```
python trainer.py \
    --config configs/user/sample.json \
    --tensorboard_tracking_folder /project/cq-training-1/project2/teams/team03/tensorboard/$USER
```

* `config`: Path to the JSON config file used to store user model, dataloader and trainer parameters and  that follows configs/user/schema.json.
* `tensorboard_tracking_folder`: Path where to store TensorBoard data and save trained model.

### Full MASS pretraining task pipeline

```
# Run pretraining task
python trainer.py \
    --config config/user/transformer_mass_v1_pretraining.json
    --tensorboard_tracking_folder /project/cq-training-1/project2/teams/team03/tensorboard/$USER
# Copy resulting model to desired location and use as source in the config for the translation task that follows
# Translation task
python trainer.py \
    --config config/user/transformer_mass_v1_translation_with_pretraining.json
    --tensorboard_tracking_folder /project/cq-training-1/project2/teams/team03/tensorboard/$USER
# For evaluation, do not forget to change the best_config in evaluator.py

# The alternating MASS training is facilitated by the script
bash tools/mass_alternate_training.sh
``` 

