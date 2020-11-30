from datetime import datetime
from calculations import find_similarity_between, show_similarity_for_images
from model_provider import ModelType, ModelProvider, IMG_SHAPE
from video_processing import VideoProcessing
import numpy as np


def testing2():
    dt_start = datetime.now()
    original_frames = VideoProcessing.extract_frames(
        path_to_video='data/Wonder_Woman_1984_Official_Main_Trailer.mov',
        img_shape=IMG_SHAPE)
    print(
        'Time elapsed for original video frames extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
    dt_start = datetime.now()
    input_frames = VideoProcessing.extract_frames(
        path_to_video='data/wonder_cut.mov',
        img_shape=IMG_SHAPE)
    print('Time elapsed for input video frames extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
    feature_extractor_model = ModelProvider.get_model()
    dt_start = datetime.now()
    original_video_features = [feature_extractor_model.predict(np.array([f])) for f in original_frames]
    print('Time elapsed for original feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
    dt_start = datetime.now()
    input_video_features = [feature_extractor_model.predict(np.array([f])) for f in input_frames]
    print('Time elapsed for input feature extraction: %s sec' % str((datetime.now() - dt_start).total_seconds()))
    dt_start = datetime.now()

    index, value = find_similarity_between(original_video_features, input_video_features)
    print(f'Max simmilarity {round(value, 2)} star from second {index}')
    # print('Original Frame {} and Input Frame {} similarity: {:.2f}%'.format(o_index, i_index, simm))
    print('Time elapsed for comparing: %s sec' % str((datetime.now() - dt_start).total_seconds()))


def testing():
    for model_type in [ModelType.resnet_v2_50]:
        feature_extractor_model = ModelProvider.get_model()
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lion.jpg', 'data/lion2.jpg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lion.jpg', 'data/men.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/dog.jpeg', 'data/men.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/men2.jpeg', 'data/men.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/women.jpeg', 'data/men.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/women.jpeg', 'data/men2.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lions1.jpg', 'data/lions2.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lion.jpg', 'data/lions2.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lion.jpg', 'data/dog.jpeg')
        show_similarity_for_images(feature_extractor_model, model_type, 'data/lions1.jpg', 'data/dog.jpeg')


if __name__ == "__main__":
    VideoProcessing()
    # testing2()
