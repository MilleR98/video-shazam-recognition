from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
from enum import Enum


class ModelType(Enum):
    efficientnet_b4 = ('efficientnet_b4', 'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1', 224, 2048)
    # for now resnet_v2_50 finding most realistic similarities
    resnet_v2_50 = ('resnet_v2_50', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4', 224, 2048)
    image_net_inception_v3 = (
        'inception_v3', 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4', 299, 2048)
    inception_resnet_v2 = (
        'inception_resnet_v2', 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4', 299, 1536)
    inaturalist_inception_v3 = (
        'inaturalist_inception_v3', 'https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4', 299, 2048)

    def __init__(self, label, url, shape_size, feature_size) -> None:
        self.LABEL = label
        self.URL = url
        self.SHAPE_SIZE = shape_size
        self.FEATURE_SIZE = feature_size


class ModelProvider:

    @staticmethod
    def get_model(model_type: ModelType, verbose: bool = True) -> tf.keras.Sequential:

        # model_path = "models/" + model_type.label + '.h5'
        # if Path(model_path).is_file():
        #    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': KerasLayer})

        model = tf.keras.Sequential(layers=[
            tf.keras.layers.InputLayer(input_shape=(model_type.SHAPE_SIZE, model_type.SHAPE_SIZE) + (3,)),
            hub.KerasLayer(model_type.URL, trainable=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(model_type.FEATURE_SIZE, activation='softmax')
        ], name='feature_extractor_based_on_%s' % model_type.LABEL)

        if verbose:
            print("Building the model")
        model.build([None, model_type.SHAPE_SIZE, model_type.SHAPE_SIZE, 3])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if verbose:
            model.summary()

        # model.save(model_path)

        return model
