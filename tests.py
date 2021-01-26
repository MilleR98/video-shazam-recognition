from cfg.globals import DATA_DIR
from processing.video_processing import VideoProcessing


def save_video():
    vp = VideoProcessing(verbose=True)

    filepaths = [pth for pth in DATA_DIR.iterdir() if pth.suffix in ('.mp4', '.mov')]

    for video_path in filepaths:
        vp.analyse_and_save(path_to_video=str(video_path))


def test_query():
    vp = VideoProcessing(verbose=True)
    vp.query_video(str(DATA_DIR / 'cuts' / 'wonder_cut.mov'))


if __name__ == "__main__":
    test_query()
