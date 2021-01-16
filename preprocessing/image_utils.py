from matplotlib import pyplot as plt
import tensorflow as tf
from numpy import ndarray


class ImageUtils:

    @staticmethod
    def show_image(image_np_array: ndarray) -> None:
        plt.imshow(image_np_array)
        plt.grid(False)
        plt.show()

    @staticmethod
    def load_and_preprocess_image(path: str, shape_size: int) -> ndarray:
        image = tf.keras.preprocessing.image.load_img(path, target_size=(shape_size, shape_size),
                                                      interpolation='bilinear')
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr /= 255.0

        return input_arr
