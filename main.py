from datetime import datetime

import numpy as np
from numpy import ndarray
from image_utils import ImageUtils
from model_provider import ModelType, ModelProvider
from video_frames_extractor import VideoFramesExtractor
import tensorflow as tf

SIMILARITY_LOWER_BOUND = 0.70
DATA_ROOT_FOLDER = 'data/'


def main():
    base_model_type = ModelType.resnet_v2_50
    target_shape = (base_model_type.SHAPE_SIZE, base_model_type.SHAPE_SIZE)

    dt_start = datetime.now()
    original_frames = VideoFramesExtractor.extract(
        path_to_video=DATA_ROOT_FOLDER + 'Wonder_Woman_1984_Official_Main_Trailer.mov',
        img_shape=target_shape)
    print(
        'Time elapsed for original video frames extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))

    dt_start = datetime.now()
    input_frames = VideoFramesExtractor.extract(
        path_to_video=DATA_ROOT_FOLDER + 'wonder_women_trailer_cut.mp4',
        img_shape=target_shape)
    print('Time elapsed for input video frames extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))

    feature_extractor_model = ModelProvider.get_model(base_model_type)

    dt_start = datetime.now()
    original_video_features = [feature_extractor_model.predict(np.array([f])) for f in original_frames]
    print('Time elapsed for original feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))

    dt_start = datetime.now()
    input_video_features = [feature_extractor_model.predict(np.array([f])) for f in input_frames]
    print('Time elapsed for input feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))

    dt_start = datetime.now()
    for o_index, o_feature in enumerate(original_video_features):
        print('Original frame #{}'.format(o_index))
        for i_index, i_feature in enumerate(input_video_features):
            simm = calculate_similarity(features1=o_feature, features2=i_feature)
            print('Original Frame {} and Input Frame {} similarity: {:.2f}%'.format(o_index, i_index, simm))
    print('Time elapsed for comparing: %s sec' % str((datetime.now() - dt_start).total_seconds()))


def testing():
    for model_type in [ModelType.resnet_v2_152]:
        feature_extractor_model = ModelProvider.get_model(model_type)

        calculate_similarity_for_images(feature_extractor_model, model_type, 'lion.jpg', 'lion2.jpg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'lion.jpg', 'men.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'dog.jpeg', 'men.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'men2.jpeg', 'men.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'women.jpeg', 'men.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'women.jpeg', 'men2.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'lions1.jpg', 'lions2.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'lion.jpg', 'lions2.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'lion.jpg', 'dog.jpeg')
        calculate_similarity_for_images(feature_extractor_model, model_type, 'lions1.jpg', 'dog.jpeg')


def calculate_similarity_for_images(feature_extractor_model, model_type, img1_path, img2_path):
    image_array1 = ImageUtils.load_and_preprocess_image(DATA_ROOT_FOLDER + img1_path, model_type.SHAPE_SIZE)
    features1 = feature_extractor_model.predict(np.array([image_array1]))
    image_array2 = ImageUtils.load_and_preprocess_image(DATA_ROOT_FOLDER + img2_path, model_type.SHAPE_SIZE)
    features2 = feature_extractor_model.predict(np.array([image_array2]))
    simm = calculate_similarity(features1, features2)

    if abs(simm) >= SIMILARITY_LOWER_BOUND:
        print('{} and {} similarity: {:.2f}%'.format(img1_path, img2_path, simm))


def calculate_similarity(features1: ndarray, features2: ndarray,
                         in_percentage: bool = True):
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_value = cosine_loss(y_true=tf.nn.l2_normalize(features1), y_pred=tf.nn.l2_normalize(features2)).numpy()

    return abs(similarity_value * 100) if in_percentage else abs(similarity_value)


if __name__ == "__main__":
    main()
