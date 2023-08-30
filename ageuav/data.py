import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import requests
import zipfile
import shutil

_HEIGHT = 256
_WIDTH = 256
_NUM_CHANNELS = 3
_NUM_IMAGES = {
    'train': 1900,
    'validation': 1900,
    'test': 1900,
}

SHUFFLE_BUFFER = _NUM_IMAGES['train']
SHAPE = [_HEIGHT, _WIDTH, _NUM_CHANNELS]


def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    #maybe_download_and_extract(data_dir)
    data_dir="/home/chathuranga_basnayaka/Desktop/my/semantic/wild/ageuav/datare"

    file_pattern = os.path.join(data_dir, "*.jpg")
    filename_dataset = tf.data.Dataset.list_files(file_pattern)

    # Check if the number of files is correct
    expected_num_files = _NUM_IMAGES['train'] if is_training else _NUM_IMAGES['validation']
    num_files = tf.data.experimental.cardinality(filename_dataset).numpy()
    if num_files != expected_num_files:
        print(f"Expected {expected_num_files} PNG files, but found {num_files} files.")
        return None

    return filename_dataset.map(lambda x: tf.image.decode_png(tf.io.read_file(x)))


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    image = tf.reshape(raw_record, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
    # normalize images to range 0-1
    image = tf.cast(image, dtype)
    image = tf.divide(image, 255.0)

    return image, image

"""
def preprocess_image(image, is_training):
    #Preprocess a single image of layout [height, width, depth]
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image"""




get_dataset(True, "data")
#parse_record()