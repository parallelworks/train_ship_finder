import json, glob, argparse, os
from datetime import datetime
from random import randint

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs

args = read_args()
max_noise = int(args['max_noise'])

np.random.seed(42)

def tf_load_img(path):
    print(datetime.now(), 'Loading: ', path, flush = True)
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img


def load_data(imgdir):
    # Find images:
    img_paths = glob.glob(os.path.join(imgdir, '*.png'))

    # Load images:
    #imgs = np.concatenate([ tf_load_img(img_path) for img_path in img_paths ])
    imgs = np.asarray([ tf_load_img(img_path) for img_path in img_paths ])
    imgs_label = [ int(os.path.basename(img_path).split('_')[0]) for img_path in img_paths ]
    return imgs, imgs_label, img_paths


def add_noise(X, max_noise = max_noise):
    return X + randint(0, max_noise) * np.random.rand(X.shape[0], X.shape[1], X.shape[2])

def str2bool(x):
    if x == 'True':
        return True
    else:
        return False


if __name__ == '__main__':

    batch_size = 100 #int(args['batch_size'])
    num_samples = int(args['num_samples'])

    # Load training data:
    imgs, imgs_label, img_paths = load_data(args['imgdir'])
    # preprocessing_function
    datagen = ImageDataGenerator(
        brightness_range = [0, float(args['max_brightness_shift'])],
        rotation_range = int(args['rotation_range']),
        horizontal_flip = str2bool(args['horizontal_flip']),
        vertical_flip = str2bool(args['vertical_flip']),
        zca_whitening = str2bool(args['zca_whitening']),
        preprocessing_function = add_noise
    )
    if str2bool(args['zca_whitening']):
        datagen.fit(imgs)

    it = datagen.flow(imgs, y = imgs_label, batch_size = batch_size)

    for i in range(int(num_samples/batch_size)):
        x, y = it.next()
        for b in range(batch_size):
            img = Image.fromarray(x[b].astype('uint8'))
            img.save(
                os.path.join(
                    args['imgdir'],
                    '{label}__generated-{it}-{batch}.png'.format(
                        label = str(y[b]),
                        it = str(i),
                        batch = str(b)
                    )
                )
            )
