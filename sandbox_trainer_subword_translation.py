import os
import json

import tensorflow as tf

from libs.data_loaders import subwords
from libs.models import transformer
from libs import helpers
# Aliasing some methods to lighten the code
TextLineDataset = tf.data.TextLineDataset

# https://www.tensorflow.org/tutorials/text/transformer

#####################################
# Data loading into TextLineDataset #
#####################################

print("Starting data loading procedure")

# For now loading data path from a config file in home folder...
cfg_name = 'ift6759_project2_sandbox_cfg.json'
with open(os.path.join(os.environ['HOME'], cfg_name)) as file_cfg:
    cfg_dict = json.loads(file_cfg.read())
path_data = cfg_dict['path_data']

# All the data files we will be using
path_trad_en_train = os.path.join(path_data, 'original', 'train.lang1.train')
path_trad_en_test = os.path.join(path_data, 'original', 'train.lang1.test')
path_unal_en_train = os.path.join(path_data, 'tokenized_default_no_punc', 'unaligned.en.train')
path_trad_en_val = os.path.join(path_data, 'original', 'train.lang1.validation')
path_unal_en_val = os.path.join(path_data, 'tokenized_default_no_punc', 'unaligned.en.validation')
path_trad_fr_train = os.path.join(path_data, 'original', 'train.lang2.train')
path_trad_fr_test = os.path.join(path_data, 'original', 'train.lang2.test')
path_unal_fr_train = os.path.join(path_data, 'tokenized_keep_case', 'unaligned.fr.train')
path_trad_fr_val = os.path.join(path_data, 'original', 'train.lang2.validation')
path_unal_fr_val = os.path.join(path_data, 'tokenized_keep_caseun'
                                           '', 'unaligned.fr.validation')
# Create all TextLineDataset objects
datasets = {
    'sentences_translation_en_train': TextLineDataset([path_trad_en_train]),
    'sentences_translation_en_test': TextLineDataset([path_trad_en_test]),
    'sentences_all_en_train': TextLineDataset([path_trad_en_train, path_unal_en_train]),
    'sentences_translation_en_validation': TextLineDataset([path_trad_en_val]),
    'sentences_all_en_validation': TextLineDataset([path_trad_en_val, path_unal_en_val]),
    'sentences_translation_fr_train': TextLineDataset([path_trad_fr_train]),
    'sentences_translation_fr_test': TextLineDataset([path_trad_fr_test]),
    'sentences_all_fr_train': TextLineDataset([path_trad_fr_train, path_unal_fr_train]),
    'sentences_translation_fr_validation': TextLineDataset([path_trad_fr_val]),
    'sentences_all_fr_validation': TextLineDataset([path_trad_fr_val, path_unal_fr_val]),
}
#
# # sentences_all_en_train = sentences_all_en_train.take(12800)
# # sentences_all_fr_train = sentences_all_fr_train.take(12800)
#
#
#
#
#
# # For now only do the translation task
# tokenizer_en = subwords.subword_tokenizer(
#     'subword_vocabulary_en', datasets['sentences_translation_en_train'])
# print(tokenizer_en.vocab_size)
# tokenizer_fr = subwords.subword_tokenizer(
#     'subword_vocabulary_fr', datasets['sentences_translation_fr_train'])
# print(tokenizer_fr.vocab_size)
#
# # Create (input, target) pairs of sentences, these are now ZipDataset
# # and the input/targets are now EagerTensor
# sentences_translation_both_train = tf.data.Dataset.zip(
#     (datasets['sentences_translation_en_train'],
#      datasets['sentences_translation_fr_train']))
# sentences_translation_both_validation = tf.data.Dataset.zip(
#     (datasets['sentences_translation_en_validation'],
#      datasets['sentences_translation_fr_validation']))
#
#
# def create_tf_encoder_bilingual(tokenizer1, tokenizer2):
#     def encode(lang1, lang2):
#         """Encode with tokenizer and add beginning/end of sentence tokens
#
#         :param lang1: sentence
#         :param lang2: sentence
#         :return: list of tokens for both languages
#         """
#         lang1 = [tokenizer1.vocab_size] + tokenizer1.encode(
#             lang1.numpy()) + [tokenizer1.vocab_size + 1]
#
#         lang2 = [tokenizer2.vocab_size] + tokenizer2.encode(
#             lang2.numpy()) + [tokenizer2.vocab_size + 1]
#
#         return lang1, lang2
#
#     def tf_py_encode(lang1, lang2):
#         result_lang1, result_lang2 = tf.py_function(
#             encode, [lang1, lang2], [tf.int64, tf.int64])
#         result_lang1.set_shape([None])
#         result_lang2.set_shape([None])
#
#         return result_lang1, result_lang2
#
#     return tf_py_encode
#
#
# tf_encode = create_tf_encoder_bilingual(tokenizer_en, tokenizer_fr)
#
#
# # MAX_LENGTH = 40
# #
# #
# # def filter_max_length(x0, y0, max_length=MAX_LENGTH):
# #     return tf.logical_and(tf.size(x0) <= max_length,
# #                           tf.size(y0) <= max_length)
#
#
# BUFFER_SIZE = 20000
# BATCH_SIZE = 64
#
# train_preprocessed = (
#     sentences_translation_both_train
#     .map(tf_encode)
#     # .filter(filter_max_length)
#     # cache the dataset to memory to get a speedup while reading from it.
#     .cache())
#     # .shuffle(BUFFER_SIZE))
#
# val_preprocessed = (
#     sentences_translation_both_validation
#     .map(tf_encode))
#     # .filter(filter_max_length))
#
# train_dataset = (train_preprocessed
#                  .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
#                  .prefetch(tf.data.experimental.AUTOTUNE))
# val_dataset = (val_preprocessed
#                .padded_batch(BATCH_SIZE,  padded_shapes=([None], [None])))
#
# # num_layers = 4
# # d_model = 128
# # dff = 512
# # num_heads = 8
# # num_layers = 3
# # d_model = 128
# # dff = 512
# # num_heads = 4  # must be a divisor of d_model
# # dropout_rate = 0.2
#
# input_vocab_size = tokenizer_en.vocab_size + 2
# target_vocab_size = tokenizer_fr.vocab_size + 2

BATCH_SIZE = 64
my_dl = subwords.SubwordDataLoader(
    helpers.load_dict('configs/user/transformer_v1.json'))
my_dl.build(BATCH_SIZE)

# transformer0 = transformer.Transformer(
#     num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
#     pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)

transformer0 = transformer.builder(
    helpers.load_dict('configs/user/transformer_v1.json'),
    my_dl.input_vocab_size, my_dl.target_vocab_size)

EPOCHS = 75

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train = False
if train:
    transformer0.fit(my_dl.training_dataset, epochs=EPOCHS)
else:
    transformer0.load_checkpoint()


def evaluate(inp_sentence):
    start_token = [my_dl.tokenizer_en.vocab_size]
    end_token = [my_dl.tokenizer_en.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + \
        my_dl.tokenizer_en.encode(inp_sentence) + \
        end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [my_dl.tokenizer_fr.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    # This limit in i should be MAX_LENGTH, but no longer using it so...
    for slen in range(100):
        enc_padding_mask, combined_mask, dec_padding_mask = \
            transformer.create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer0(encoder_input,
                                                      output,
                                                      False,
                                                      enc_padding_mask,
                                                      combined_mask,
                                                      dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == my_dl.tokenizer_fr.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the
        # decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = my_dl.tokenizer_fr.decode(
        [j for j in result if j < my_dl.tokenizer_fr.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    return predicted_sentence


# all test set:
test_en, test_fr = (datasets['sentences_translation_en_test'],
                    datasets['sentences_translation_fr_test'])
with open('tmp_results.txt', 'w') as file_out:
    # hack to reduce nb of predictions
    with open('tmp_targets.txt', 'w') as file_targets:
        for i, (eng_sent, fr_target) in enumerate(zip(test_en, test_fr)):
            if i > 20:
                break
            file_targets.write(fr_target.numpy().decode())
            file_targets.write('\n')
            file_out.write(translate(eng_sent.numpy()))
            file_out.write('\n')
