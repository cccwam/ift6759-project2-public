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

    class CustomTFBert(TFBertPreTrainedModel):
        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)

#            self.embeddings = TFBertEmbeddings(config, name="embeddings")
#            self.encoder = TFBertEncoder(config, name="encoder")
#            self.decoder = TFBertEncoder(config, name="encoder")
#            self.pooler = TFBertPooler(config, name="pooler")

            self.bert_encoder = TFBertMainLayer(configuration_encoder, name="encoder_bert")
            self.bert_decoder = TFBertMainLayer(configuration_decoder, name="decoder_bert")
            self.mlm = TFBertMLMHead(configuration_decoder, self.bert_decoder.embeddings, name="mlm___cls")

        def get_output_embeddings(self):
            return self.bert_encoder.embeddings

        @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
        def call(self, inputs, **kwargs):
            r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            prediction_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
                tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
                tuple of :obj:`tf.Tensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            import tensorflow as tf
            from transformers import BertTokenizer, TFBertForMaskedLM

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
            input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
            outputs = model(input_ids)
            prediction_scores = outputs[0]

            """
            outputs = self.bert_encoder(inputs, **kwargs) # TODO here

            sequence_output = outputs[0]
            self.bert_decoder(inputs=None, inputs_embeds=sequence_output)
            prediction_scores = self.mlm(sequence_output, training=kwargs.get("training", False))

            outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

            return outputs  # prediction_scores, (hidden_states), (attentions)
    # Idea for using Bert as part of larger model
    # https://github.com/huggingface/transformers/issues/1350
    if "pretrained_model_huggingface" in model_hparams:
        pretrained_model_huggingface_path: str = model_hparams["pretrained_model_huggingface"]
        logger.info(
            f"Load existing model from {pretrained_model_huggingface_path}")  # TODO for decoder, force is_decoder=true
        bert_model_encoder = TFBertForMaskedLM.from_pretrained(pretrained_model_huggingface_path)
        raise NotImplementedError()
        #        bert_model_decoder:TFBertForMaskedLM = TFBertForMaskedLM(configuration_decoder, is_decoder=True)
    else:
        logger.info(f"No existing models")
        # Initializing models from the configuration
#        bert_model_encoder = CustomTFBert(configuration_decoder, name="encoder_decoder")
        bert_model_encoder = TFBertMainLayer(configuration_encoder, name="encoder")
        bert_model_encoder.embeddings = TFBertEmbeddings(configuration_encoder, name="encoder_embeddings")
        bert_model_encoder.encoder = TFBertEncoder(configuration_encoder, name="encoder_encoder")
        bert_model_encoder.pooler = TFBertPooler(configuration_encoder, name="encoder_pooler")

        bert_model_decoder = TFBertMainLayer(configuration_decoder, name="decoder")
        bert_mlm = TFBertMLMHead(configuration_decoder, bert_model_decoder.embeddings, name="mlm___cls")

    token_inputs = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                         name="BERT_token_inputs")
    attention_masks = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                            name="BERT_attention_masks")
    token_type_ids = tf.keras.layers.Input(shape=(seq_length_source,), dtype=tf.int32,
                                           name="BERT_token_token_type_ids")
#    prediction_scores, _ = bert_model_encoder([token_inputs, attention_masks, token_type_ids])
    outputs = bert_model_encoder([token_inputs, attention_masks, token_type_ids])
    last_hidden_states = outputs[0]
    outputs = bert_model_decoder(None, inputs_embeds=last_hidden_states,
                                 token_type_ids=token_type_ids)
    last_hidden_states = outputs[0]

    prediction_scores = bert_mlm(last_hidden_states)

    model = tf.keras.Model([token_inputs, attention_masks, token_type_ids], prediction_scores, name=name)

    model.summary(line_length=120, print_fn=logger.info)
    return model
