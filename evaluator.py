import argparse
import subprocess
import tempfile


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
    import tensorflow as tf

    from libs import helpers
    from libs.data_loaders.abstract_dataloader import AbstractDataloader
    from libs.models import transformer

    import tqdm

    import logging
    from libs.data_loaders.abstract_dataloader import create_masks_fm
    from libs.data_loaders.dataloader_bilingual_huggingface import BilingualTranslationHFSubword
    from libs.data_loaders.dataloader_bilingual_tensorflow import BilingualTranslationTFSubword
    from libs.data_loaders.mass_subword import MassSubwordDataLoader
    from libs.models.transformer import Encoder, Decoder

    logger = tf.get_logger()
    logger.setLevel(logging.DEBUG)

    # best_config_file = '/project/cq-training-1/project2/teams/team03/models/transformer_mass_v1_translation_with_pretraining_resume.json'
    best_config_file = 'configs/user/transformers-fm/TFM_TINY_BBPE_eval_fm.json'
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
    batch_size = 32  # TODO should be 128 on Helios
    data_loader.build(batch_size=batch_size)
    test_dataset = data_loader.test_dataset

    all_predictions = []
    if isinstance(data_loader, MassSubwordDataLoader):
        all_predictions = transformer.inference(
            data_loader.tokenizer, model, test_dataset)
    else:
        if isinstance(data_loader, BilingualTranslationTFSubword) or \
                isinstance(data_loader, BilingualTranslationHFSubword):
            sample_to_display = 10

            encoder: Encoder = model.get_layer("encoder")
            decoder: Decoder = model.get_layer("decoder")
            final_layer: tf.keras.layers.Dense = model.layers[-1]

            for inputs, mask in tqdm.tqdm(test_dataset, total=data_loader.test_steps):

                mini_batch_size = inputs.shape[0]
                dec_inp = tf.Variable(tf.zeros((mini_batch_size, data_loader.get_seq_length() + 1), dtype=tf.int32))

                bos_tensor = tf.convert_to_tensor(data_loader.bos)
                bos_tensor = tf.reshape(bos_tensor, [1, 1])
                bos_tensor = tf.tile(bos_tensor, multiples=[mini_batch_size, 1])

                dec_inp[:, 0].assign(bos_tensor[:, 0])  # BOS token

                # WARNING: IF THE MODEL USED WAS FROM A TF FILE, A LOT OF WARNINGS WILL APPEAR
                #  Workaround: Use the hdf5 format to load the final model
                # https://github.com/tensorflow/tensorflow/issues/35146
                def get_preds(encoder, decoder, final_layer, dec_inp, inputs, mask, max_seq):
                    enc_output: tf.Tensor = encoder.__call__(inputs=inputs, mask=mask, training=False)

                    for timestep in range(max_seq):
                        _, combined_mask, dec_padding_mask = create_masks_fm(inp=inputs, tar=dec_inp[:, :-1])

                        dec_output, attention_weights = decoder(
                            inputs=dec_inp[:, :-1], enc_output=enc_output, look_ahead_mask=combined_mask,
                            padding_mask=dec_padding_mask)

                        outputs = final_layer(inputs=dec_output)  # (batch_size, seq_length, vocab_size)
                        pred = tf.argmax(outputs[:, timestep, :], axis=-1)
                        pred = tf.cast(pred, dtype=tf.int32)
                        dec_inp[:, timestep + 1].assign(pred)
                    return dec_inp

                predictions = get_preds(
                    encoder=encoder,
                    decoder=decoder,
                    final_layer=final_layer,
                    dec_inp=dec_inp,
                    inputs=inputs,
                    mask=mask,
                    # TODO Decision to be made, 100 seq length doesn't seem to hurt perfs
                    max_seq=100)  # data_loader.get_seq_length())
                for prediction in predictions.numpy():
                    if sample_to_display > 0:
                        logger.info(f"Example of generated translation: {data_loader.decode(prediction)}")
                        sample_to_display -= 1
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
    print(out.stdout)
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
