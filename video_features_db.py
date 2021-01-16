import pickle
from dataclasses import dataclass
from typing import List

from bson import Binary
from pymongo import MongoClient


def npArray2Binary(npArray):
    """Utility method to turn an numpy array into a BSON Binary string.
    utilizes pickle protocol 2 (see http://www.python.org/dev/peps/pep-0307/
    for more details).
    Called by stashNPArrays.
    :param npArray: numpy array of arbitrary dimension or a list of npArrays
    :returns: BSON Binary object a pickled numpy array (or a list).
    """

    if type(npArray) is list:
        return [Binary(pickle.dumps(f_vec)) for f_vec in npArray]

    return Binary(pickle.dumps(npArray, protocol=2), subtype=128)


def binary2npArray(binary):
    """Utility method to turn a a pickled numpy array string back into
    a numpy array.
    Called by loadNPArrays, and thus by loadFullData and loadFullExperiment.
    :param binary: BSON Binary object a pickled numpy array or a list of objects.
    :returns: numpy array of arbitrary dimension (or a list)
    """

    if type(binary) is list:
        return [pickle.loads(b_vec) for b_vec in binary]

    return pickle.loads(binary)


@dataclass
class VideoFeatures(dict):
    name: str
    feature_vectors: list
    original_video_url: str
    _id: str = None


class VideoFeaturesDb:
    _COLLECTION_NAME: str = 'VideoFeatures'
    _DB_NAME: str = 'VideosDB'

    def __init__(self) -> None:
        mongo_client = MongoClient(host='localhost', port=27017, document_class=dict)
        self._db = mongo_client[self._DB_NAME]

    def get_all_video_features(self) -> List[VideoFeatures]:
        fetch_result = self._db[self._COLLECTION_NAME].find()

        return [
            VideoFeatures(
                name=persistent_video['name'],
                feature_vectors=binary2npArray(persistent_video['feature_vectors']),
                original_video_url=persistent_video['original_video_url'],
                _id=persistent_video['_id']
            )
            for persistent_video in fetch_result
        ]

    def get_video_features_by_name(self, search_name: str) -> VideoFeatures:
        persistent_video = self._db[self._COLLECTION_NAME].find_one(filter={'name': search_name})

        return VideoFeatures(
            name=persistent_video['name'],
            feature_vectors=binary2npArray(persistent_video['feature_vectors']),
            original_video_url=persistent_video['original_video_url'],
            _id=persistent_video['_id']
        )

    def save_processed_video(self, video_features: VideoFeatures):
        dict_values = video_features.__dict__
        del dict_values['_id']
        dict_values['feature_vectors'] = npArray2Binary(dict_values['feature_vectors'])
        self._db[self._COLLECTION_NAME].insert_one(video_features.__dict__)

    def update_processed_video(self, video_features: VideoFeatures):
        dict_values = video_features.__dict__
        dict_values['feature_vectors'] = npArray2Binary(dict_values['feature_vectors'])
        self._db[self._COLLECTION_NAME].update_one(video_features.__dict__)
