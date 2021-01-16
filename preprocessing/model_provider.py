import tensorflow as tf
import tensorflow_hub as hub
from enum import Enum

from logger_wrapper import Log, LogLevel


class ModelType(Enum):
    efficientnet_b4 = ('efficientnet_b4', 'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1', 224, 2048)
    efficientnet_b7 = ('efficientnet_b7', 'https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1', 600, 2048)
    # for now resnet_v2_50 finding most realistic similarities
    resnet_v2_50 = ('resnet_v2_50', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4', 224, 2048)
    resnet_v2_152 = ('resnet_v2_152', 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4', 224, 2048)
    image_net_inception_v3 = (
        'inception_v3', 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4', 299, 2048)
    inaturalist_inception_v3 = (
        'inaturalist_inception_v3', 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4', 299, 2048)

    def __init__(self, label, url, shape_size, feature_size) -> None:
        self.LABEL = label
        self.URL = url
        self.SHAPE_SIZE = shape_size
        self.FEATURE_SIZE = feature_size


BASE_MODEL_TYPE = ModelType.resnet_v2_50
IMG_SHAPE = (BASE_MODEL_TYPE.SHAPE_SIZE, BASE_MODEL_TYPE.SHAPE_SIZE)

_log = Log(level=LogLevel.DEBUG)


class ModelProvider:
    FEATURE_EXTRACTING_MODEL = None

    @classmethod
    def instance(cls):
        if cls.FEATURE_EXTRACTING_MODEL is None:
            cls.FEATURE_EXTRACTING_MODEL = cls.build_model()

        return cls.FEATURE_EXTRACTING_MODEL

    @classmethod
    def build_model(cls, base_model_type: ModelType = None, verbose: bool = True) -> tf.keras.Sequential:

        if base_model_type is None:
            base_model_type = BASE_MODEL_TYPE

        model = tf.keras.Sequential(layers=[
            hub.KerasLayer(base_model_type.URL, trainable=False),
        ], name='feature_extractor_based_on_%s' % base_model_type.LABEL)

        _log.info("Building the model")
        model.build([None, base_model_type.SHAPE_SIZE, base_model_type.SHAPE_SIZE, 3])

        if verbose:
            _log.debug("Model summary:")
            model.summary()

        return model
