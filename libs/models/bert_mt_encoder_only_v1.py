import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM

logger = tf.get_logger()

"""
    BERT model for the translation task and translation language model task.

    THIS IS THE MODEL USED FOR THE PROJECT 2 OF IFT6759
"""


def builder(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]
    model_hparams = config["model"]["hyper_params"]

    vocab_size_source = dl_hparams["vocab_size_source"]
    seq_length_source = dl_hparams["seq_length_source"]

    vocab_size_target = dl_hparams["vocab_size_target"]
    seq_length_target = dl_hparams["seq_length_target"]

    assert vocab_size_source == vocab_size_target, "vocab_size_source and vocab_size_target must be the same for BERT"

    consolidated_seq_length = seq_length_source + seq_length_target

    # For the config see https://github.com/google-research/bert
    name = model_hparams["name"]
    num_hidden_layers = model_hparams["num_hidden_layers"]
    num_attention_heads = model_hparams["num_attention_heads"]
    hidden_size = model_hparams["hidden_size"]

    configuration = BertConfig(vocab_size=vocab_size_source,
                               num_hidden_layers=num_hidden_layers,
                               num_attention_heads=num_attention_heads,
                               max_position_embeddings=consolidated_seq_length,
                               hidden_size=hidden_size)

    # Idea for using Bert as part of larger model
    # https://github.com/huggingface/transformers/issues/1350
    if "pretrained_model_huggingface" in model_hparams:
        pretrained_model_huggingface_path: str = model_hparams["pretrained_model_huggingface"]
        logger.info(f"Load existing model from {pretrained_model_huggingface_path}")
        bert_model = TFBertForMaskedLM.from_pretrained(pretrained_model_huggingface_path)
    else:
        logger.info(f"No existing model")
        # Initializing a model from the configuration
        bert_model = TFBertForMaskedLM(configuration)

    token_inputs = tf.keras.layers.Input(shape=(consolidated_seq_length,), dtype=tf.int32,
                                         name="BERT_token_inputs")
    attention_masks = tf.keras.layers.Input(shape=(consolidated_seq_length,), dtype=tf.int32,
                                            name="BERT_attention_masks")
    token_type_ids = tf.keras.layers.Input(shape=(consolidated_seq_length,), dtype=tf.int32,
                                           name="BERT_token_token_type_ids")
    outputs = bert_model([token_inputs, attention_masks, token_type_ids])

    model = tf.keras.Model([token_inputs, attention_masks, token_type_ids], outputs, name=name)

    model.summary(line_length=120, print_fn=logger.info)

    return model
