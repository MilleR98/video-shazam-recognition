from Katna.video import Video
from matplotlib import pyplot as plt

from cfg.globals import DATA_DIR
from processing.video_features_db import VideoFeaturesDb
from processing.video_processing import VideoProcessing


def analyse_and_save():
    vp = VideoProcessing(verbose=True, video_db=VideoFeaturesDb())

    filepaths = [pth for pth in DATA_DIR.iterdir() if pth.suffix in ('.mp4', '.mov')]

    for index, video_path in enumerate(filepaths):
        if index == 500:
            break
        vp.analyse_and_save(path_to_video=str(video_path))


def test_query():
    vp = VideoProcessing(verbose=True)
    vp.query_video(str(DATA_DIR / 'cuts' / 'wonder_cut.mov'))


def show_images(images) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)


def kframe_test():
    vd = Video()
    from datetime import datetime
    dt_start = datetime.now()
    test_video_path = '/Users/omelnyk/Work/vidiscovery-recognition/data/KALUSH feat alyona alyona  Гори_v720P.mp4'
    duration = vd._get_video_duration_with_ffmpeg(test_video_path)
    print(int(duration))
    imgs = vd.extract_video_keyframes(no_of_frames=int(duration), file_path=test_video_path)
    print(len(imgs))
    print("extraction time: " + str((datetime.now() - dt_start).total_seconds()))
    _, axs = plt.subplots(4, 5, figsize=(20, 20))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    analyse_and_save()
