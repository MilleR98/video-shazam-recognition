import operator
from datetime import datetime

import numpy as np
import faiss

from cfg.globals import DATA_DIR
from preprocessing.model_provider import ModelProvider, IMG_SHAPE
from processing.video_features_db import VideoFeaturesDb
from processing.video_processing import VideoProcessing


def init_tree():
    index_tree = faiss.index_factory(2048, 'IVF1000, Flat')

    print(index_tree.is_trained)

    video_map = {}
    i = 0

    vectors_list = []
    for j in range(0, 2):
        print(f'Iter: {j}')
        for video in video_db.get_all_video_features():
            for feature_vector in video.feature_vectors:
                video_map[i] = video.name
                vectors_list.append(feature_vector[0])
                i = i + 1


    vectors = np.array(vectors_list).astype('float32')
    index_tree.train(vectors)  # Train на нашем наборе векторов

    # Обучение завершено, но векторов в индексе пока нет, так что добавляем их в индекс:
    print(index_tree.is_trained)  # True
    print(index_tree.ntotal)  # 0
    index_tree.add(vectors)
    print(index_tree.ntotal)

    return index_tree, video_map


if __name__ == '__main__':

    video_db = VideoFeaturesDb()

    index_tree, video_map = init_tree()

    feature_extractor = ModelProvider.instance()

    def search(fragment_path):

        dt_start = datetime.now()

        input_frames, duration = VideoProcessing.extract_frames(
            path_to_video=str(fragment_path),
            img_shape=IMG_SHAPE)
        input_video_features = [feature_extractor.predict(np.array([f])) for f in input_frames]
        unique_nns = {}
        values = []
        for i_v in input_video_features:
            dist, indexes = index_tree.search(i_v, 1)
            values.extend(dist[0].tolist())
            for n, index in enumerate(indexes[0].tolist()):
                v_name = video_map[index]
                if v_name not in unique_nns:
                    unique_nns[v_name] = 0
                else:
                    unique_nns[v_name] = unique_nns[v_name] + 1


        avg_simm = sum(values) / len(values)
        print(f"Fragment: {video_path}")
        print(f'Target movie: {(max(unique_nns.items(), key=operator.itemgetter(1))[0]) if avg_simm > 30 else "NOT FOUND"}')
        print(f'Time elapsed {str((datetime.now() - dt_start).total_seconds())}')
        print(f'Average simm: {avg_simm} %')
        print('------------------###--------------')

    filepaths = [pth for pth in (DATA_DIR / 'cuts').iterdir() if pth.suffix in ('.mp4', '.mov')]

    for video_path in filepaths:
        search(video_path)