import cv2


class VideoFramesExtractor:

    def get_frame(self, video_cap: cv2.VideoCapture, sec: int, img_name: str):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, image = video_cap.read()
        if has_frames:
            cv2.imwrite(img_name + ".jpg", image)  # save frame as JPG file

        return has_frames

    def extract(self, path_to_video: str, frame_rate: int = 1):
        video_cap = cv2.VideoCapture(path_to_video)

        sec = 0
        count = 1
        success = self.get_frame(video_cap, sec, "image" + str(count))
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success = self.get_frame(video_cap, sec, "image" + str(count))

        video_cap.release()
        cv2.destroyAllWindows()
