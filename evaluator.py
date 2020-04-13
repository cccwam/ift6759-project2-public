import argparse
import logging
import subprocess
import tempfile

import tqdm
from numpy import int32

from libs.data_loaders.dataloader_bilingual_tensorflow import BilingualTranslationTFSubword
from libs.models.transformer import Encoder, Decoder


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).
    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.
    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.
    Returns: None
    """

    ##### MODIFY BELOW #####
    import numpy as np
    import tensorflow as tf

    from libs import helpers
    from libs.data_loaders.abstract_dataloader import AbstractDataloader
    from libs.models import transformer

    logger = tf.get_logger()
    logger.setLevel(logging.DEBUG)

    best_config_file = '/project/cq-training-1/project2/teams/team03/models/transformer_mass_v1_translation_with_pretraining_resume.json'
    # best_config_file = 'configs/user/transformers-fm/TFM_TINY_TF_eval_fm.json'
    logger.info(f"Using best config file: {best_config_file}")
    best_config = helpers.load_dict(best_config_file)
    helpers.validate_user_config(best_config)

    # TODO: Edit our AbstractDataloader to support a raw_english_test_set_file_path. Currently it only supports
    #   preprocessed data defined directly in best_config.
    data_loader: AbstractDataloader = helpers.get_online_data_loader(config=best_config,
                                                                     raw_english_test_set_file_path=input_file_path)

    if best_config["model"]["definition"]["module"] == 'libs.models.transformerv2':
        model = transformer.load_transformer(best_config)
    else:
        mirrored_strategy = helpers.get_mirrored_strategy()
        if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
            with mirrored_strategy.scope():
                model: tf.keras.Model = helpers.prepare_model(config=best_config)
        else:
            model: tf.keras.Model = helpers.prepare_model(config=best_config)

#    batch_size = 32  # 32 is max for 6GB GPU memory
    batch_size = 128  # TODO to check if ok with GPU
    data_loader.build(batch_size=batch_size)
    test_dataset = data_loader.test_dataset

    all_predictions = []
    if best_config["data_loader"]["definition"]["name"] == 'MassSubwordDataLoader':
        all_predictions = transformer.inference(
            data_loader.tokenizer, model, test_dataset)
    else:
        # TODO to test again
        if isinstance(data_loader, BilingualTranslationTFSubword):
            encoder: Encoder = model.get_layer("encoder")
            decoder: Decoder = model.get_layer("decoder")
            final_layer: tf.keras.layers.Dense = model.layers[-1]

            for inputs, mask in test_dataset:

                mini_batch_size = inputs.shape[0]
                enc_output: tf.Tensor = encoder.__call__(inputs=inputs, mask=mask, training=False)

                dec_inp = np.zeros((mini_batch_size, data_loader.get_seq_length() + 1), dtype=int32)
                dec_inp[:, 0] = data_loader.bos  # BOS token
                # TODO not necesarly with HF tokenizer

                for timestep in tqdm.tqdm(range(data_loader.get_seq_length())):
                    _, combined_mask, dec_padding_mask = data_loader.create_masks(
                        inp=inputs,
                        tar=dec_inp[:, :-1]
                    )

                    dec_output, attention_weights = decoder(
                        inputs=dec_inp[:, :-1], enc_output=enc_output, look_ahead_mask=combined_mask,
                        padding_mask=dec_padding_mask)

                    outputs = final_layer(inputs=dec_output)  # (batch_size, seq_length, vocab_size)

                    dec_inp[:, timestep + 1] = tf.argmax(outputs[:, timestep, :], axis=-1).numpy()

                predictions = dec_inp

                if isinstance(predictions, tf.Tensor):
                    predictions = predictions.numpy()
                for prediction in predictions:
                    all_predictions += [data_loader.decode(prediction)]

        else:
            raise NotImplementedError(f"No method to generate for class {data_loader.__class__.__name__}")

    with open(pred_file_path, 'w+') as file_handler:
        for prediction in all_predictions:
            file_handler.write(f'{prediction}\n')

    ##### MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
