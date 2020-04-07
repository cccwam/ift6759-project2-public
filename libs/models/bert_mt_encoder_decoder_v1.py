import typing

import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM, TFBertModel, TFBertMainLayer, TFXLMRobertaForMaskedLM, \
    TFPreTrainedModel, TFBertPreTrainedModel, TFBertEmbeddings
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_tf_bert import TFBertEncoder, TFBertMLMHead, BERT_INPUTS_DOCSTRING, TFBertPooler

from libs.models.bert_mlm_v1 import builder_mlm

logger = tf.get_logger()

"""
    BERT model for the translation task and translation language model task.
    It's Encoder-Decoder Transformer only

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

    # For the config see https://github.com/google-research/bert
    name = model_hparams["name"]
    num_hidden_layers = model_hparams["num_hidden_layers"]
    num_attention_heads = model_hparams["num_attention_heads"]
    hidden_size = model_hparams["hidden_size"]

    # TODO clean

    configuration_encoder = BertConfig(vocab_size=vocab_size_source,
                                       num_hidden_layers=num_hidden_layers,
                                       num_attention_heads=num_attention_heads,
                                       max_position_embeddings=seq_length_source,
                                       output_hidden_states=True,
                                       hidden_size=hidden_size)
    configuration_decoder = BertConfig(vocab_size=vocab_size_target,
                                       num_hidden_layers=num_hidden_layers,
                                       num_attention_heads=num_attention_heads,
                                       max_position_embeddings=seq_length_target,
                                       is_decoder=True,
                                       hidden_size=hidden_size)


    # Idea for using Bert as part of larger model
    # https://github.com/huggingface/transformers/issues/1350
    if "pretrained_model_huggingface" in model_hparams:
        pretrained_model_huggingface_path: str = model_hparams["pretrained_model_huggingface"]
        logger.info(
            f"Load existing model from {pretrained_model_huggingface_path}")  # TODO for decoder, force is_decoder=true
        bert_model_encoder = TFBertModel.from_pretrained(pretrained_model_huggingface_path) # Better perf
#        bert_model_decoder = TFBertModel.from_pretrained(pretrained_model_huggingface_path) Similar perf
        bert_model_decoder = TFBertModel(configuration_decoder, name="decoder")
#        bert_model_decoder.trainable = False # Does not help
        bert_mlm = TFBertMLMHead(configuration_decoder, bert_model_decoder.bert.embeddings, name="mlm")
    else:
        logger.info(f"No existing models")
        # Initializing models from the configuration
        bert_model_encoder = TFBertModel(configuration_encoder, name="encoder")
        bert_model_decoder = TFBertModel(configuration_decoder, name="decoder")
        bert_mlm = TFBertMLMHead(configuration_decoder, bert_model_decoder.bert.embeddings, name="mlm")

    token_inputs = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                         name="BERT_token_inputs")
    attention_masks = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                            name="BERT_attention_masks")
    token_type_ids = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                           name="BERT_token_token_type_ids")

    outputs = bert_model_encoder([token_inputs, attention_masks, token_type_ids])
    last_hidden_states = outputs[0]

    dummy_token_type_ids = tf.ones((seq_length_target,), dtype=tf.int32)
    outputs = bert_model_decoder(None, inputs_embeds=last_hidden_states,
                                 token_type_ids=dummy_token_type_ids # Dummy input to prevent
                                 )
    last_hidden_states = outputs[0]

    prediction_scores = bert_mlm(last_hidden_states)

    model = tf.keras.Model([token_inputs, attention_masks, token_type_ids], prediction_scores, name=name)

    model.summary(line_length=120, print_fn=logger.info)
    return model
