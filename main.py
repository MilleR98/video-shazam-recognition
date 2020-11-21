import numpy as np
from tensorflow.python.keras.models import Model

from image_utils import ImageUtils
from model_provider import ModelType, ModelProvider
from video_frames_extractor import VideoFramesExtractor
import tensorflow as tf

# video_extractor = VideoFramesExtractor()
# video_extractor.extract('data/lake_in_park.MOV')

SIMILARITY_LOWER_BOUND = 0.6
DATA_ROOT_FOLDER = 'data/'


def main():
    for model_type in ModelType:
        feature_extractor_model = ModelProvider.get_model(model_type)
        testing(feature_extractor_model, model_type)


def testing(feature_extractor_model, model_type):
    calculate_similarity(feature_extractor_model, model_type, 'lion.jpg', 'lion2.jpg')
    calculate_similarity(feature_extractor_model, model_type, 'lion.jpg', 'men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'dog.jpeg', 'men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'men2.jpeg', 'men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'women.jpeg', 'men.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'women.jpeg', 'men2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'lions1.jpg', 'lions2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'lion.jpg', 'lions2.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'lion.jpg', 'dog.jpeg')
    calculate_similarity(feature_extractor_model, model_type, 'lions1.jpg', 'dog.jpeg')


def calculate_similarity(feature_extractor_model: Model, model_type: ModelType, image_path1: str, image_path2: str):
    image_array1 = ImageUtils.load_and_preprocess_image(DATA_ROOT_FOLDER + image_path1, model_type.SHAPE_SIZE)
    image_array2 = ImageUtils.load_and_preprocess_image(DATA_ROOT_FOLDER + image_path2, model_type.SHAPE_SIZE)
    # ImageUtils.show_image(image_array)
    features1 = feature_extractor_model.predict(np.array([image_array1]))
    features2 = feature_extractor_model.predict(np.array([image_array2]))
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_value = cosine_loss(y_true=tf.nn.l2_normalize(features1), y_pred=tf.nn.l2_normalize(features2)).numpy()
    if abs(similarity_value) >= SIMILARITY_LOWER_BOUND:
        print('{} and {} similarity: {:.2f}%'.format(image_path1, image_path2, abs(similarity_value * 100)))


if __name__ == "__main__":
    main()
