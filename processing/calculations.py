import operator
from typing import Tuple, List

import numpy as np
import tensorflow as tf

SIMILARITY_LOWER_BOUND = 65
MAX_SUCCESSIVELY_MISMATCH = 3


def calculate_similarity(features1: np.ndarray, features2: np.ndarray, in_percentage: bool = True) -> float:
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_value = cosine_loss(y_true=tf.nn.l2_normalize(features1), y_pred=tf.nn.l2_normalize(features2)).numpy()
    similarity_value = round(similarity_value, 4)

    return round(abs(similarity_value * 100) if in_percentage else abs(similarity_value), 4)


def find_similarity_between(original_video_features: List[np.ndarray], input_video_features: List[np.ndarray]) \
        -> Tuple[int, float]:
    steps_simm = []
    max_steps = len(original_video_features) - len(input_video_features)

    for window_step in range(0, max_steps):
        steps_simm.append([])
        successively_mismatch_counter = 0
        window_offset = 1
        for i_index, i_feature in enumerate(input_video_features):
            simm = calculate_similarity(original_video_features[i_index + window_step], i_feature)
            successively_mismatch_counter = successively_mismatch_counter + 1 if simm < SIMILARITY_LOWER_BOUND else 0
            steps_simm[window_step].append(simm)

            if successively_mismatch_counter > MAX_SUCCESSIVELY_MISMATCH:
                window_offset = MAX_SUCCESSIVELY_MISMATCH
                break
        window_step += window_offset

    steps_avg_simm = [(sum(simm_values) / len(input_video_features)) for simm_values in steps_simm]
    index, result_simm_value = max(enumerate(steps_avg_simm), key=operator.itemgetter(1))

    return index, round(result_simm_value, 2)
