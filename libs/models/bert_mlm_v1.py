import typing

from libs.models.bert_mt_encoder_only_v1 import builder

logger = tf.get_logger()

"""
    BERT model for a masked language model task for one language only.

    It's the same model as machine translation task.
"""


def builder_mlm(
        config: typing.Dict[typing.AnyStr, typing.Any]):
    dl_hparams = config["data_loader"]["hyper_params"]

    config["data_loader"]["hyper_params"]["vocab_size_source"] = dl_hparams["vocab_size"]
    config["data_loader"]["hyper_params"]["vocab_size_target"] = dl_hparams["vocab_size"]
    config["data_loader"]["hyper_params"]["seq_length_source"] = dl_hparams["seq_length"]
    config["data_loader"]["hyper_params"]["seq_length_target"] = 0

    return builder(config=config)
