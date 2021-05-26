import json

from pytube import YouTube

YOUTUBE_VIDEO_URL_TEMPLATE = 'https://www.youtube.com/watch?v={}'


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


YDL_OPTS = {
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
}


def download_video(youtube_id: str) -> str:
    try:
        yt = YouTube(YOUTUBE_VIDEO_URL_TEMPLATE.format(youtube_id))
        return yt.streams.filter(file_extension='mp4') \
            .first() \
            .download(filename=yt.title, output_path='tmp')
    except Exception as ex:
        print(ex)


def download_all_videos(limit):
    with open('trailer.urls.unique30K.v1.json') as f:
        data = json.load(f)
        for index, item in enumerate(data):
            print(index)
            if index > 1500:
                break
            if 1000 < index < 1500:
                video_file_path: str = download_video(item['youtube_id'])
