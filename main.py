from pathlib import Path

from logger_wrapper import Log
from video_processing import VideoProcessing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CFG_DIR = BASE_DIR / "cfg"
LOG_CONF_FILE = CFG_DIR / "logger_config.yml"

Log.configure(path_to_config=LOG_CONF_FILE, root_dir_path=BASE_DIR)


def save_video():
    vp = VideoProcessing(verbose=True)
    vp.analyse_and_save(
        path_to_video='/Users/omelnyk/Work/video-shazam-recognition/data/Wonder_Woman_1984_Official_Main_Trailer.mov')


def test_query():
    vp = VideoProcessing(verbose=True)
    vp.query_video('/Users/omelnyk/Work/video-shazam-recognition/data/wonder_cut_start.mov')


if __name__ == "__main__":
    test_query()
