import os
from datetime import datetime
from typing import Tuple, List

import cv2
import numpy as np

from calculations import find_similarity_between
from model_provider import ModelProvider, IMG_SHAPE
from video_features_db import VideoFeaturesDb, VideoFeatures

SUPPORTED_VIDEO_TYPES = ('.mov', 'mp4')


def validate_path(path_to_video):
    if not os.path.exists(path_to_video) or not os.path.isfile(path_to_video):
        raise ValueError('Invalid path to video file')
    if not path_to_video.endswith(SUPPORTED_VIDEO_TYPES):
        raise ValueError(f'Invalid file format... expected: {SUPPORTED_VIDEO_TYPES}')


class VideoProcessing:

    def __init__(self) -> None:
        self._video_db = VideoFeaturesDb()

    def analyse_and_save(self, path_to_video: str):

        validate_path(path_to_video)

        video_frames = self.extract_frames(path_to_video=path_to_video, img_shape=IMG_SHAPE)
        feature_extractor_model = ModelProvider.get_model()
        original_video_features = [feature_extractor_model.predict(np.array([f])) for f in video_frames]

        video_features_info = VideoFeatures(name=str(os.path.basename(path_to_video)),
                                            feature_vectors=original_video_features,
                                            original_video_url=os.path.dirname(path_to_video))

        self._video_db.save_processed_video(video_features_info)

    def query_video(self, path_to_fragment: str):
        validate_path(path_to_fragment)

        print(f'Input video {os.path.basename(path_to_fragment)}')

        dt_start = datetime.now()
        input_frames = VideoProcessing.extract_frames(path_to_video=path_to_fragment, img_shape=IMG_SHAPE)
        print('Time elapsed for input video frames extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
        feature_extractor_model = ModelProvider.get_model()
        input_video_features = [feature_extractor_model.predict(np.array([f])) for f in input_frames]
        print('Time elapsed for input feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
        dt_start = datetime.now()

        db_video_infos: List[VideoFeatures] = self._video_db.get_all_video_features()

        current_max = 0.
        for vid_info in db_video_infos:
            print(f'Checking video {vid_info.name}')
            original_video_features = vid_info.feature_vectors
            index, value = find_similarity_between(original_video_features, input_video_features)
            if current_max > value:
                current_max = value
            print(f'Max simmilarity {round(value, 2)} star from second {index}')

        print('Time elapsed for comparing: %s sec' % str((datetime.now() - dt_start).total_seconds()))

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

        video_cap.release()
        cv2.destroyAllWindows()

        return frames
