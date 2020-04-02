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
    import numpy as np
    import tensorflow as tf

    from libs import helpers
    from libs.data_loaders.abstract_dataloader import AbstractDataloader

    # best_config = 'configs/user/lm_lstm_fr_v1.json'
    best_config_file = 'configs/user/transformer_subword_tutorial_v1_translation_with_eng_pretraining.local.json'
    print(f"Using best config file: {best_config_file}")
    best_config = helpers.load_dict(best_config_file)
    # ToDo make sure others don't use this
    if 'pretrained_layers' in best_config["model"]["hyper_params"]:
        del best_config["model"]["hyper_params"]["pretrained_layers"]
    helpers.validate_user_config(best_config)

    # TODO: Edit our AbstractDataloader to support a raw_english_test_set_file_path. Currently it only supports
    #   preprocessed data defined directly in best_config.
    data_loader: AbstractDataloader = helpers.get_online_data_loader(best_config, input_file_path)
    model: tf.keras.Model = helpers.get_model(best_config)

    batch_size = 64
    data_loader.build(batch_size)
    test_dataset = data_loader.test_dataset

    all_predictions = []
    # ToDo better logic for using alternate data loader
    if best_config["data_loader"]["definition"]["name"] == 'SubwordDataLoader':
        for i, eng_sent in enumerate(test_dataset):
            # ToDo better verbosity
            print(i)
            start_token = [data_loader.tokenizer_en.vocab_size]
            end_token = [data_loader.tokenizer_en.vocab_size + 1]
            inp_sentence = start_token + \
                           data_loader.tokenizer_en.encode(eng_sent.numpy()) + \
                           end_token
            encoder_input = tf.expand_dims(inp_sentence, 0)

            # as the target is english, the first word to the transformer should be the
            # english start token.
            decoder_input = [data_loader.tokenizer_fr.vocab_size]
            transformer_output = tf.expand_dims(decoder_input, 0)
            end_token = data_loader.tokenizer_fr.vocab_size + 1
            result, attention_weights = model.evaluate(
                encoder_input, transformer_output, end_token)
            predicted_sentence = data_loader.tokenizer_fr.decode(
                [j for j in result if j < data_loader.tokenizer_fr.vocab_size])
            all_predictions.append(predicted_sentence)
    else:
        for mini_batch in test_dataset.batch(batch_size):
            # Outputs are mini_batch[:, 1]
            model_inputs = mini_batch[:, 0]
            # TODO: Use the dataloader's decode function to get a list of tokens instead of text before calling predict()
            predictions = model.predict(model_inputs)
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            all_predictions.append(predictions)
        all_predictions = np.concatenate(all_predictions, axis=0)

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
