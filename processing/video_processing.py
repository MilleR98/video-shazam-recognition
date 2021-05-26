import operator
import os
from datetime import datetime
from typing import Tuple, List

import cv2
import faiss
import numpy as np

from logger_wrapper import Log, LogLevel
from preprocessing.model_provider import ModelProvider, IMG_SHAPE
from processing.calculations import find_similarity_between, SIMILARITY_LOWER_BOUND
from processing.video_features_db import VideoFeaturesDb, VideoFeatures

SUPPORTED_VIDEO_TYPES = ('.mov', 'mp4')


def validate_path(path_to_video):
    if not os.path.exists(path_to_video) or not os.path.isfile(path_to_video):
        raise ValueError('Invalid path to video file')
    if not path_to_video.endswith(SUPPORTED_VIDEO_TYPES):
        raise ValueError(f'Invalid file format... expected: {SUPPORTED_VIDEO_TYPES}')


class VideoProcessing:

    def __init__(self, verbose: bool = False, video_db: VideoFeaturesDb = None) -> None:
        if video_db is None:
            self._video_db = VideoFeaturesDb()
        else:
            self._video_db = video_db
        self._log = Log(level=LogLevel.DEBUG if verbose else None)

        self.frame_feature_index = faiss.index_factory(2048, 'IVF1000, Flat')

        self.video_map = {}
        i = 0

        vectors_list = []
        for video in video_db.get_all_video_features():
            for feature_vector in video.feature_vectors:
                self.video_map[i] = video.name
                vectors_list.append(feature_vector[0])
                i = i + 1

        vectors = np.array(vectors_list).astype('float32')
        self.frame_feature_index.train(vectors)
        self.frame_feature_index.add(vectors)

    def analyse_and_save(self, path_to_video: str):

        validate_path(path_to_video)

        self._log.info(f'Original Video to process {os.path.basename(path_to_video)}')

        dt_start = datetime.now()
        video_frames, duration = self.extract_frames(path_to_video=path_to_video, img_shape=IMG_SHAPE)
        self._log.debug('Time elapsed for original video frames extraction: %s sec'
                        % str((datetime.now() - dt_start).total_seconds()))

        feature_extractor_model = ModelProvider.instance()

        dt_start = datetime.now()
        original_video_features = [feature_extractor_model.predict(np.array([f])) for f in video_frames]
        video_features_info = VideoFeatures(name=os.path.splitext(str(os.path.basename(path_to_video)))[0],
                                            feature_vectors=original_video_features,
                                            original_video_url=path_to_video,
                                            duration=duration)
        self._log.debug('Time elapsed for original feature extraction: %s sec'
                        % str((datetime.now() - dt_start).total_seconds()))

        self._video_db.save_processed_video(video_features_info)

    def query_video_v2(self, path_to_fragment: str):

        feature_extractor_model = ModelProvider.instance()

        dt_start = datetime.now()

        input_frames, duration = VideoProcessing.extract_frames(
            path_to_video=str(path_to_fragment),
            img_shape=IMG_SHAPE)
        input_video_features = [feature_extractor_model.predict(np.array([f])) for f in input_frames]
        unique_nns = {}
        values = []
        for i_v in input_video_features:
            dist, indexes = self.frame_feature_index.search(i_v, 1)
            values.extend(dist[0].tolist())
            for n, index in enumerate(indexes[0].tolist()):
                v_name = self.video_map[index]
                if v_name not in unique_nns:
                    unique_nns[v_name] = 0
                else:
                    unique_nns[v_name] = unique_nns[v_name] + 1

        avg_simm = sum(values) / len(values)
        print(f'Target movie: {(max(unique_nns.items(), key=operator.itemgetter(1))[0]) if avg_simm > 50 else "NOT FOUND"}')
        print(f'Time elapsed {str((datetime.now() - dt_start).total_seconds())}')
        print(f'Average simm: {avg_simm} %')
        print('------------------###--------------')

        original_video_name = (max(unique_nns.items(), key=operator.itemgetter(1))[0]) if avg_simm > 50 else None
        original_video_url = self._video_db.get_video_features_by_name(original_video_name).original_video_url if original_video_name is not None else None

        return {
            'isFound': original_video_name is not None,
            'name': original_video_name,
            'elapsedTime': str((datetime.now() - dt_start).total_seconds()),
            'video_full_url': original_video_url
        }

    def query_video(self, path_to_fragment: str):
        validate_path(path_to_fragment)

        self._log.info(f'Input video {os.path.basename(path_to_fragment)}')

        query_dt_start = datetime.now()

        dt_start = datetime.now()
        input_frames, duration = VideoProcessing.extract_frames(path_to_video=path_to_fragment, img_shape=IMG_SHAPE)
        self._log.debug('Time elapsed for input video frames extraction: %s sec'
                        % str((datetime.now() - dt_start).total_seconds()))

        feature_extractor_model = ModelProvider.instance()

        dt_start = datetime.now()
        input_video_features = [feature_extractor_model.predict(np.array([f])) for f in input_frames]
        self._log.debug('Time elapsed for input feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))

        db_video_infos: List[VideoFeatures] = self._video_db.get_all_video_features()

        current_max = SIMILARITY_LOWER_BOUND
        original_video_name = None
        original_video_url = None
        for vid_info in db_video_infos:
            self._log.info(f'Checking video {vid_info.name}')
            original_video_features = vid_info.feature_vectors
            index, value = find_similarity_between(original_video_features, input_video_features)
            if value > current_max:
                current_max = value
                original_video_name = vid_info.name
                original_video_url = vid_info.original_video_url
            self._log.info(f'Max similarity {round(value, 2)} star from second {index}')

        self._log.debug('Time elapsed for comparing: %s sec' % str((datetime.now() - dt_start).total_seconds()))

        query_dt_end = datetime.now()

        return {
            'isFound': original_video_name is not None,
            'name': original_video_name,
            'elapsedTime': (query_dt_end - query_dt_start).total_seconds(),
            'video_full_url': original_video_url
        }

    @staticmethod
    def __get_frame(video_cap: cv2.VideoCapture, sec: int, img_shape: Tuple[int, int]):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, frame_img = video_cap.read()
        if has_frames:
            frame_img = cv2.resize(frame_img, img_shape, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        return has_frames, frame_img

    @staticmethod
    def extract_frames(path_to_video: str, img_shape: Tuple[int, int] = IMG_SHAPE, normalize: bool = True, frame_rate: int = 1):
        video_cap = cv2.VideoCapture(path_to_video, apiPreference=cv2.CAP_FFMPEG)

        sec = 0
        count = 1
        success = VideoProcessing.__get_frame(video_cap, sec, img_shape)
        frames = []
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success, frame_img = VideoProcessing.__get_frame(video_cap, sec, img_shape)

            if success:
                if normalize:
                    frame_img = frame_img.astype(float)
                    frame_img /= 255.0
                frames.append(frame_img)

        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count / fps)

        video_cap.release()
        cv2.destroyAllWindows()

        return frames, duration

    @staticmethod
    def extract_key_frames(path_to_video, number_of_):
        pass
