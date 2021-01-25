from cfg.globals import DATA_DIR
from processing.video_processing import VideoProcessing


def save_video():
    vp = VideoProcessing(verbose=True)
    vp.analyse_and_save(
        path_to_video=str(DATA_DIR / 'Wonder_Woman_1984_Official_Main_Trailer.mov'))


def test_query():
    vp = VideoProcessing(verbose=True)
    vp.query_video(str(DATA_DIR / 'wonder_cut_start.mov'))


if __name__ == "__main__":
    save_video()
    test_query()
