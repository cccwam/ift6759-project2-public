import tensorflow as tf

from libs.models.transformer import Encoder, Decoder

logger = tf.get_logger()


def load_pretrained_layers(config: dict, my_model: tf.keras.Model):
    if "pretrained_layers" in config["model"]["hyper_params"]:
        pretrained_layers = config["model"]["hyper_params"]["pretrained_layers"]
        for pretrained_layer in pretrained_layers:
            logger.info(
                f"Load pretrained layer {pretrained_layer['layer_name']} into {pretrained_layer['target_layer_name']}")
            # https://github.com/tensorflow/tensorflow/issues/32348
            pretrained_model: tf.keras.Model \
                = tf.keras.models.load_model(pretrained_layer["model_path"], compile=False)

            if (pretrained_layer['target_layer_name'] == "decoder") and \
                    isinstance(my_model.get_layer("decoder"), Decoder):
                logger.debug(f"Load a Transformer Encoder into Transformer Decoder")
                pretrained_encoder: Encoder = pretrained_model.get_layer("encoder")
                decoder: Decoder = my_model.get_layer("decoder")

                # Load weights from encoder like XLM paper
                # See https://github.com/facebookresearch/XLM/blob/master/src/model/transformer.py

                decoder.embedding.set_weights(pretrained_encoder.embedding.get_weights())
                for i in range(len(decoder.dec_layers)):
                    decoder.dec_layers[i].mha2.set_weights(pretrained_encoder.enc_layers[i].mha.get_weights())
                    decoder.dec_layers[i].ffn.set_weights(pretrained_encoder.enc_layers[i].ffn.get_weights())
                    decoder.dec_layers[i].layernorm2.set_weights(
                        pretrained_encoder.enc_layers[i].layernorm2.get_weights())
                    decoder.dec_layers[i].layernorm3.set_weights(
                        pretrained_encoder.enc_layers[i].layernorm2.get_weights())

            else:
                for l in pretrained_layer['layer_name'].split("/"):
                    pretrained_model = pretrained_model.get_layer(l)

                weights = pretrained_model.get_weights()
                my_model.get_layer(pretrained_layer['target_layer_name']).set_weights(weights)

    return my_model
