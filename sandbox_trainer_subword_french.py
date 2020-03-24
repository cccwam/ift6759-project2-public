import os
import json
import time

import tensorflow as tf

from libs.data_loaders import subwords
from libs.models import transformer
# Aliasing some methods to lighten the code
TextLineDataset = tf.data.TextLineDataset

# https://www.tensorflow.org/tutorials/text/transformer

#####################################
# Data loading into TextLineDataset #
#####################################

# For now loading data path from a config file in home folder...
cfg_name = 'ift6759_project2_sandbox_cfg.json'
with open(os.path.join(os.environ['HOME'], cfg_name)) as file_cfg:
    cfg_dict = json.loads(file_cfg.read())
path_data = cfg_dict['path_data']

# All the data files we will be using
path_trad_en_train = os.path.join(path_data, 'original', 'train.lang1.train')
path_unal_en_train = os.path.join(path_data, 'tokenized_default_no_punc', 'unaligned.en.train')
path_trad_en_val = os.path.join(path_data, 'original', 'train.lang1.validation')
path_unal_en_val = os.path.join(path_data, 'tokenized_default_no_punc', 'unaligned.en.validation')
path_trad_fr_train = os.path.join(path_data, 'original', 'train.lang2.train')
path_unal_fr_train = os.path.join(path_data, 'tokenized_keep_case', 'unaligned.fr.train')
path_trad_fr_val = os.path.join(path_data, 'original', 'train.lang2.validation')
path_unal_fr_val = os.path.join(path_data, 'tokenized_keep_case', 'unaligned.fr.validation')
# Create all TextLineDataset objects
datasets = {
    'sentences_translation_en_train': TextLineDataset([path_trad_en_train]),
    'sentences_all_en_train': TextLineDataset([path_trad_en_train, path_unal_en_train]),
    'sentences_translation_en_validation': TextLineDataset([path_trad_en_val]),
    'sentences_all_en_validation': TextLineDataset([path_trad_en_val, path_unal_en_val]),
    'sentences_translation_fr_train': TextLineDataset([path_trad_fr_train]),
    'sentences_all_fr_train': TextLineDataset([path_trad_fr_train, path_unal_fr_train]),
    'sentences_translation_fr_validation': TextLineDataset([path_trad_fr_val]),
    'sentences_all_fr_validation': TextLineDataset([path_trad_fr_val, path_unal_fr_val]),
}

# sentences_all_en_train = sentences_all_en_train.take(12800)
# sentences_all_fr_train = sentences_all_fr_train.take(12800)

tokenizer_fr = subwords.subword_tokenizer(
    'subword_vocabulary_fr_02', datasets['sentences_all_fr_train'])
print(type(tokenizer_fr))

# Create (input, target) pairs of sentences, these are now ZipDataset
# and the input/targets are now EagerTensor
sentences_all_both_train = tf.data.Dataset.zip(
    (datasets['sentences_all_fr_train'],
     datasets['sentences_all_fr_train']))
sentences_all_both_validation = tf.data.Dataset.zip(
    (datasets['sentences_all_fr_validation'],
     datasets['sentences_all_fr_validation']))


def create_tf_encoder_bilingual(tokenizer1, tokenizer2):
    def encode(lang1, lang2):
        """Encode with tokenizer and add beginning/end of sentence tokens

        :param lang1: sentence
        :param lang2: sentence
        :return: list of tokens for both languages
        """
        lang1 = [tokenizer1.vocab_size] + tokenizer1.encode(
            lang1.numpy()) + [tokenizer1.vocab_size + 1]

        lang2 = [tokenizer2.vocab_size] + tokenizer2.encode(
            lang2.numpy()) + [tokenizer2.vocab_size + 1]

        return lang1, lang2

    def tf_py_encode(lang1, lang2):
        result_lang1, result_lang2 = tf.py_function(
            encode, [lang1, lang2], [tf.int64, tf.int64])
        result_lang1.set_shape([None])
        result_lang2.set_shape([None])

        return result_lang1, result_lang2

    return tf_py_encode


tf_encode = create_tf_encoder_bilingual(tokenizer_fr, tokenizer_fr)


# MAX_LENGTH = 40
#
#
# def filter_max_length(x0, y0, max_length=MAX_LENGTH):
#     return tf.logical_and(tf.size(x0) <= max_length,
#                           tf.size(y0) <= max_length)


BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_preprocessed = (
    sentences_all_both_train
    .map(tf_encode)
    # .filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    .cache())
    # .shuffle(BUFFER_SIZE))

val_preprocessed = (
    sentences_all_both_validation
    .map(tf_encode))
    # .filter(filter_max_length))

train_dataset = (train_preprocessed
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
                 .prefetch(tf.data.experimental.AUTOTUNE))
val_dataset = (val_preprocessed
               .padded_batch(BATCH_SIZE,  padded_shapes=([None], [None])))

# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
num_layers = 3
d_model = 128
dff = 512
num_heads = 4  # must be a divisor of d_model
dropout_rate = 0.2

input_vocab_size = tokenizer_fr.vocab_size + 2
target_vocab_size = tokenizer_fr.vocab_size + 2

learning_rate = transformer.CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer0 = transformer.HalfTransformer(
    num_layers, d_model, num_heads, dff, input_vocab_size,
    pe_input=input_vocab_size, rate=dropout_rate)

checkpoint_path = "../checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer0,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

EPOCHS = 1

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = inp[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = \
        transformer.create_masks(tar_inp, tar_real)

    with tf.GradientTape() as tape:
        predictions, _ = transformer0(tar_inp,
                                      True,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer0.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer0.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = [tokenizer_fr.vocab_size]
    end_token = [tokenizer_fr.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    # inp_sentence = start_token + tokenizer_fr.encode(inp_sentence) + end_token
    # encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_fr.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    # This limit in i should be MAX_LENGTH, but no longer using it so...
    for i in range(1000):
        enc_padding_mask, combined_mask, dec_padding_mask = \
            transformer.create_masks(output, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer0(output,
                                                      False,
                                                      enc_padding_mask,
                                                      combined_mask,
                                                      dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_fr.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_fr.decode([i for i in result
                                              if i < tokenizer_fr.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))


translate("La privatisation totale , le désengagement de l' État et des collectivités publiques n' est alors pas un progrès , mais une régression .")
print("Real translation: La privatisation totale , le désengagement de l' État et des collectivités publiques n' est alors pas un progrès , mais une régression .")
