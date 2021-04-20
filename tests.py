from cfg.globals import DATA_DIR
from preprocessing.image_utils import ImageUtils
from processing.video_features_db import VideoFeaturesDb
from processing.video_processing import VideoProcessing
from Katna.video import Video
from matplotlib import pyplot as plt

def save_video():
    vp = VideoProcessing(verbose=True)

    filepaths = [pth for pth in DATA_DIR.iterdir() if pth.suffix in ('.mp4', '.mov')]

    for video_path in filepaths:
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
    imgs = vd.extract_video_keyframes(no_of_frames=20, file_path='/Users/omelnyk/Work/vidiscovery-recognition/data/Політехніка на карантині.mp4')

    _, axs = plt.subplots(4, 5, figsize=(20, 20))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()

if __name__ == "__main__":
    kframe_test()

