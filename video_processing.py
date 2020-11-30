from typing import Tuple

import cv2
import numpy as np

from model_provider import ModelProvider, IMG_SHAPE
from video_features_db import VideoFeaturesDb, VideoFeatures


class VideoProcessing:

    def __init__(self) -> None:
        self._video_db = VideoFeaturesDb()

    def analyse_and_save(self, video, video_url=None):
        video_frames = self.extract_frames(path_to_video=video, img_shape=IMG_SHAPE)
        feature_extractor_model = ModelProvider.get_model()
        original_video_features = [feature_extractor_model.predict(np.array([f])) for f in video_frames]

        video_features_info = VideoFeatures(name=str(video), feature_vectors=original_video_features, original_video_url=video_url)
        self._video_db.save_processed_video(video_features_info)

    @staticmethod
    def __get_frame(video_cap: cv2.VideoCapture, sec: int, img_shape: Tuple[int, int], img_name: str = ''):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, frame_img = video_cap.read()
        if has_frames:
            frame_img = cv2.resize(frame_img, img_shape, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        return has_frames, frame_img

    @staticmethod
    def extract_frames(path_to_video, img_shape: Tuple[int, int] = IMG_SHAPE, normalize: bool = True, frame_rate: int = 1):
        video_cap = cv2.VideoCapture(path_to_video, apiPreference=cv2.CAP_FFMPEG)

        sec = 0
        count = 1
        success = VideoProcessing.__get_frame(video_cap, sec, img_shape, "image" + str(count))
        frames = []
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success, frame_img = VideoProcessing.__get_frame(video_cap, sec, img_shape, "image" + str(count))

            if success:
                if normalize:
                    frame_img = frame_img.astype(float)
                    frame_img /= 255.0
                frames.append(frame_img)

        video_cap.release()
        cv2.destroyAllWindows()

        return frames
