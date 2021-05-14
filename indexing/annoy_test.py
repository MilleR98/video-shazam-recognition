import operator
import pickle
from datetime import datetime

import annoy as ann
import numpy as np

from cfg.globals import DATA_DIR
from preprocessing.model_provider import ModelProvider, IMG_SHAPE
from processing.video_features_db import VideoFeaturesDb
from processing.video_processing import VideoProcessing


def init_tree():
    index_tree = ann.AnnoyIndex(2048, 'angular')

    video_map = {}
    i = 0

    for j in range(0, 1):
        print(f'Iter: {j}')
        for video in video_db.get_all_video_features():
            for feature_vector in video.feature_vectors:
                video_map[i] = video.name
                index_tree.add_item(i, feature_vector[0])
                i = i + 1

    index_tree.build(20)

    index_tree.save('test.ann')

    with open('index_tree.pickle', 'wb') as handle:
        pickle.dump(video_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Index counter: {i}')

    return index_tree, video_map


def load_dumped_tree():
    with open('index_tree.pickle', 'rb') as handle:
        b = pickle.load(handle)

    u = ann.AnnoyIndex(2048, 'angular')
    u.load('test.ann')

    return u, b



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
            indexes, dist = index_tree.get_nns_by_vector(i_v[0], 1, include_distances=True)
            values.extend(dist)
            for n, index in enumerate(indexes):
                v_name = video_map[index]
                if v_name not in unique_nns:
                    unique_nns[v_name] = 0
                else:
                    unique_nns[v_name] = unique_nns[v_name] + 1

        avg_simm = 100 - ((sum(values) / len(values)) * 100)
        print(f"Fragment: {video_path}")
        print(f'Target movie: {(max(unique_nns.items(), key=operator.itemgetter(1))[0]) if avg_simm > 30 else "NOT FOUND"}')
        print(f'Time elapsed {str((datetime.now() - dt_start).total_seconds())}')
        print(f'Average simm: {avg_simm} %')
        print('------------------###--------------')

    filepaths = [pth for pth in (DATA_DIR / 'cuts').iterdir() if pth.suffix in ('.mp4', '.mov')]

    for video_path in filepaths:
        search(video_path)
