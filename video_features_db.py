import pickle
from dataclasses import dataclass
from typing import List

from bson import Binary
from pymongo import MongoClient


@dataclass
class VideoFeatures(dict):
    name: str
    feature_vectors: list
    original_video_url: str
    id: str = None


class VideoFeaturesDb:
    _COLLECTION_NAME: str = 'VideoFeatures'
    _DB_NAME: str = 'VideosDB'

    def __init__(self) -> None:
        mongo_client = MongoClient(host='localhost', port=27017, document_class=dict)
        self._db = mongo_client[self._DB_NAME]

    def get_all_video_features(self) -> List[VideoFeatures]:
        fetch_result = self._db[self._COLLECTION_NAME].find()
        videos = []
        for video in fetch_result:
            video['feature_vectors'] = [pickle.loads(f) for f in video['feature_vectors']]
            videos.append(video)
        return videos

    def get_video_features_by_name(self, search_name: str) -> VideoFeatures:
        fetch_result = self._db[self._COLLECTION_NAME].find_one(filter={'name': search_name})
        fetch_result['feature_vectors'] = [pickle.loads(f) for f in fetch_result['feature_vectors']]
        return fetch_result

    def save_processed_video(self, video_features: VideoFeatures):
        dict_values = video_features.__dict__
        dict_values['feature_vectors'] = [Binary(pickle.dumps(f)) for f in dict_values['feature_vectors']]
        self._db[self._COLLECTION_NAME].insert_one(video_features.__dict__)

    def update_processed_video(self, video_features: VideoFeatures):
        dict_values = video_features.__dict__
        dict_values['feature_vectors'] = [Binary(pickle.dumps(f)) for f in dict_values['feature_vectors']]
        self._db[self._COLLECTION_NAME].update_one(video_features.__dict__)
