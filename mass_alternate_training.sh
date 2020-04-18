#!/bin/bash
# This script runs the alternating strategy for MASS pretraining.
# An initial run of the translation task is required with
# python trainer.py -c=configs/user/transformer_mass_v1_translation_no_pretraining.json
# and the resulting model must be copied, in this case we use the location:
# /project/cq-training-1/project2/teams/team03/models/libs.models.transformerv2.MASS_V1_TRANSLATION_ROTATION_TAKE2.hdf5
# make sure no other models of the form libs.models.* are currently in the directory
for i in {1..10}
do
    python trainer.py -c=configs/user/transformer_mass_v1_pretraining_resume.json | tail -n3 >> mass_log.txt
    MODEL_CP=$(ls libs.models.*)
    cp $MODEL_CP /project/cq-training-1/project2/teams/team03/models/libs.models.transformerv2.MASS_V1_TRANSLATION_ROTATION_TAKE2.hdf5
    rm $MODEL_CP
    python trainer.py -c=configs/user/transformer_mass_v1_translation_with_pretraining_resume.json | tail -n3 >> mass_log.txt
    MODEL_CP=$(ls libs.models.*)
    cp $MODEL_CP /project/cq-training-1/project2/teams/team03/models/libs.models.transformerv2.MASS_V1_TRANSLATION_ROTATION_TAKE2.hdf5
    rm $MODEL_CP
    # The evaluator line requires the proper config file reference in evaluator.py
    # python evaluator.py --target-file-path /project/cq-training-1/project2/teams/team03/data/split_01/original/train.lang2.validation --input-file-path /project/cq-training-1/project2/teams/team03/data/split_01/original/train.lang1.validation | tail -n3 >> mass_log.txt
done
