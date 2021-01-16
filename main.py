from video_processing import VideoProcessing


def save_video():
    vp = VideoProcessing()
    vp.analyse_and_save(
        path_to_video='/Users/omelnyk/Work/video-shazam-recognition/data/Wonder_Woman_1984_Official_Main_Trailer.mov')


def test_query():
    vp = VideoProcessing()
    vp.query_video('/Users/omelnyk/Work/video-shazam-recognition/data/wonder_cut_start.mov')


if __name__ == "__main__":
    test_query()
