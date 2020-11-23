from typing import Tuple

import cv2


class VideoFramesExtractor:

    @staticmethod
    def __get_frame(video_cap: cv2.VideoCapture, sec: int, img_shape: Tuple[int, int], img_name: str = ''):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, frame_img = video_cap.read()
        if has_frames:
            frame_img = cv2.resize(frame_img, img_shape, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        return has_frames, frame_img

    @staticmethod
    def extract(path_to_video: str, img_shape: Tuple[int, int], normalize: bool = True, frame_rate: int = 1):
        video_cap = cv2.VideoCapture(path_to_video, apiPreference=cv2.CAP_FFMPEG)

        sec = 0
        count = 1
        success = VideoFramesExtractor.__get_frame(video_cap, sec, img_shape, "image" + str(count))
        frames = []
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success, frame_img = VideoFramesExtractor.__get_frame(video_cap, sec, img_shape, "image" + str(count))

            if success:
                if normalize:
                    frame_img = frame_img.astype(float)
                    frame_img /= 255.0
                frames.append(frame_img)

        video_cap.release()
        cv2.destroyAllWindows()

        return frames
