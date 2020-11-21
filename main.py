import numpy as np
from tensorflow.python.keras.models import Model

from image_utils import ImageUtils
from model_provider import ModelType, ModelProvider
from video_frames_extractor import VideoFramesExtractor
import tensorflow as tf

# video_extractor = VideoFramesExtractor()
# video_extractor.extract('data/lake_in_park.MOV')

SIMILARITY_LOWER_BOUND = 0.6


def main():
    for model_type in [ModelType.efficientnet_b4, ModelType.resnet_v2_50, ModelType.image_net_inception_v3,
                       ModelType.inception_resnet_v2, ModelType.inaturalist_inception_v3]:
        feature_extractor_model = ModelProvider.get_model(model_type)

        testing(feature_extractor_model, model_type)


def testing(feature_extractor_model, model_type):
    calculate_similarity(feature_extractor_model, model_type, 'data/lion.jpg', 'data/lion2.jpg')
    calculate_similarity(feature_extractor_model, model_type, 'data/lion.jpg', 'data/men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/dog.jpeg', 'data/men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/men2.jpeg', 'data/men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/women.jpeg', 'data/men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/women.jpeg', 'data/men2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/lions1.jpg', 'data/lions2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/lion.jpg', 'data/lions2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/lion.jpg', 'data/dog.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'data/lions1.jpg', 'data/dog.jpeg')


def calculate_similarity(feature_extractor_model: Model, model_type: ModelType, image_path1: str, image_path2: str):
    image_array1 = ImageUtils.load_and_preprocess_image(image_path1, model_type.SHAPE_SIZE)
    image_array2 = ImageUtils.load_and_preprocess_image(image_path2, model_type.SHAPE_SIZE)
    # ImageUtils.show_image(image_array)
    features1 = feature_extractor_model.predict(np.array([image_array1]))
    features2 = feature_extractor_model.predict(np.array([image_array2]))
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_value = cosine_loss(y_true=tf.nn.l2_normalize(features1), y_pred=tf.nn.l2_normalize(features2)).numpy()
    if abs(similarity_value) >= SIMILARITY_LOWER_BOUND:
        print('{} and {} similarity: {}'.format(image_path1, image_path2, similarity_value))


if __name__ == "__main__":
    main()
