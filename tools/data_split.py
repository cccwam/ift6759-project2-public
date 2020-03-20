import os
import argparse
import json
import random


def count_lines(input_paths):
    n_lines = None
    for input_path in input_paths:
        nl = 0
        with open(input_path, 'r') as input_file:
            for _ in input_file:
                nl += 1
        if n_lines is None:
            n_lines = nl
        else:
            if nl != n_lines:
                raise NotImplementedError(
                    "Grouped files must have same number of lines."
                )
    return n_lines


def pending1(lang_split):
    n_lines = count_lines(lang_split['input_files'])
    random.seed(lang_split['seed'])
    sentences = []
    for i, input_path in enumerate(lang_split['input_files']):
        ns = 0
        with open(input_path, 'r') as input_file:
            for sentence in input_file:
                if i == 0:
                    sentences.append([sentence])
                else:
                    sentences[ns].append(sentence)
                ns += 1
    random.shuffle(sentences)
    s1 = int((n_lines * lang_split['split'][0]) / 100.0)
    s2 = int(
        (n_lines * (lang_split['split'][0] + lang_split['split'][1])) / 100.0)
    for i, output_path in enumerate(lang_split['output_files']):
        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_train_file = open(output_path + '.train', 'w')
        output_validation_file = open(output_path + '.validation', 'w')
        output_test_file = open(output_path + '.test', 'w')
        for ns in range(n_lines):
            if ns < s1:
                output_train_file.write(sentences[ns][i])
            elif ns < s2:
                output_validation_file.write(sentences[ns][i])
            else:
                output_test_file.write(sentences[ns][i])
        output_train_file.close()
        output_validation_file.close()
        output_test_file.close()


def main(config_path):
    with open(config_path, 'r') as file_config:
        split_cfg = json.loads(file_config.read())
    for lang_split in split_cfg:
        pending1(lang_split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        help='path to the pandas catalog file')
    args = parser.parse_args()
    main(args.config_path)
