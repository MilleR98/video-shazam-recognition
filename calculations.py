import operator
from typing import Tuple, List
import tensorflow as tf
import numpy as np

from image_utils import ImageUtils

SIMILARITY_LOWER_BOUND = 0.75


def calculate_similarity(features1: np.ndarray, features2: np.ndarray, in_percentage: bool = True) -> float:
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_value = cosine_loss(y_true=tf.nn.l2_normalize(features1), y_pred=tf.nn.l2_normalize(features2)).numpy()
    similarity_value = round(similarity_value, 4)

    return round(abs(similarity_value * 100) if in_percentage else abs(similarity_value), 4)


def show_similarity_for_images(feature_extractor_model, model_type, img1_path, img2_path) -> None:
    image_array1 = ImageUtils.load_and_preprocess_image(img1_path, model_type.SHAPE_SIZE)
    features1 = feature_extractor_model.predict(np.array([image_array1]))
    image_array2 = ImageUtils.load_and_preprocess_image(img2_path, model_type.SHAPE_SIZE)
    features2 = feature_extractor_model.predict(np.array([image_array2]))
    simm_value = calculate_similarity(features1, features2)

    if abs(simm_value) >= SIMILARITY_LOWER_BOUND:
        print('{} and {} similarity: {:.4f}%'.format(img1_path, img2_path, simm_value))


def find_similarity_between(original_video_features: List[np.ndarray], input_video_features: List[np.ndarray]) -> Tuple[int, float]:
    steps_simm = []
    max_steps = len(original_video_features) - len(input_video_features)

    for window_step in range(0, max_steps):
        steps_simm.append([])
        for i_index, i_feature in enumerate(input_video_features):
            simm = calculate_similarity(original_video_features[i_index + window_step], i_feature)
            if simm < SIMILARITY_LOWER_BOUND:
                break
            steps_simm[window_step].append(simm)
        window_step += 1

    steps_avg_simm = [(sum(simm_values) / len(input_video_features)) for simm_values in steps_simm]
    index, result_simm_value = max(enumerate(steps_avg_simm), key=operator.itemgetter(1))

    return index, round(result_simm_value, 2)
