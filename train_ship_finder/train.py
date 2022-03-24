import json, glob, argparse, os
from datetime import datetime
import itertools

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs

np.random.seed(42)

mirrored_strategy = tf.distribute.MirroredStrategy(devices = ["GPU:0"])

def define_model():
    # network design
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

def compile_model(model, learning_rate, momentum):
    # optimization setup
     sgd = SGD(learning_rate = learning_rate, momentum = momentum, nesterov=True)
     model.compile(
         loss='categorical_crossentropy',
         optimizer=sgd,
         metrics=['accuracy']
     )
     return model


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

    height = imgs.shape[1]
    width = imgs.shape[1]

    # output encoding
    y = np_utils.to_categorical(imgs_label, 2)

    # Shuffle all indexes:
    indexes = np.arange(imgs.shape[0])
    np.random.shuffle(indexes)

    imgs = imgs[indexes] / 255
    y = y[indexes]
    img_paths = [ img_paths[i] for i in indexes ]

    # Split data in train, validation and test:
    X_train, X_test, Y_train, Y_test, paths_train, paths_test = train_test_split(imgs, y, img_paths)
    X_train, X_valid, Y_train, Y_valid, paths_train, paths_valid = train_test_split(X_train, Y_train, paths_train)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, X_valid, Y_valid, paths_valid

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, path = ''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi = 500)

# Create dataframe for design explorer
def create_dex_df(Y, Y_pred, data_set, paths, min_ship_score):
    Y_pred_class = np.argmax( (Y_pred > min_ship_score), axis = 1)
    Y_class = np.argmax( (Y > min_ship_score), axis = 1)

    df = pd.DataFrame(
        {
            'in:Data-Set': [ data_set for i in Y_class ],
            'in:True-Class': [ 'Ship' if y > 0.5 else 'No Ship' for y in Y_class ],
            'out:Predicted-Class': [ 'Ship' if y > 0.5 else 'No Ship' for y in Y_pred_class ],
            'out:Score': Y_pred[:, 1],
            'img:Image': paths
        }
    )
    return df


if __name__ == '__main__':
    args = read_args()
    min_ship_score = float(args['min_ship_score'])

    os.makedirs(args['model_dir'], exist_ok = True)

    # Load training data:
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, X_valid, Y_valid, paths_valid = load_data(args['imgdir'])

    # Train:
    with mirrored_strategy.scope():
        model = define_model()
        model = compile_model(model, float(args['learning_rate']), float(args['momentum']))

    early_stopping_cb = keras.callbacks.EarlyStopping(patience = int(args['patience']), restore_best_weights = True)

    history = model.fit(
        X_train,
        Y_train,
        batch_size = int(args['batch_size']), # 32 photos at once
        epochs = int(args['epochs']),
        validation_data = (X_valid, Y_valid),
        callbacks = [early_stopping_cb],
        shuffle = True,
        verbose = 2
    )
    model.save(args['model_dir'])

    # Save training history:
    hist_pd = pd.DataFrame(history.history)
    hist_pd.to_csv(os.path.join(args['model_dir'], 'history.csv'), index = False)
    hist_plot = hist_pd.plot()
    hist_plot.figure.savefig(os.path.join(args['model_dir'], 'history.png'), dpi = 500)

    # Validate model:
    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)
    Y_test_pred = model.predict(X_test)

    Y_test_pred_class = np.argmax( (Y_test_pred > min_ship_score), axis = 1)
    Y_test_class = np.argmax( (Y_test > min_ship_score), axis = 1)

    confusion_mtx = confusion_matrix(Y_test_class, Y_test_pred_class)
    plot_confusion_matrix(confusion_mtx, classes = ['No Ship', 'Ship'], path = os.path.join(args['model_dir'], 'confusion_matrix.png'))

    # Design explorer:
    df_dex = pd.concat(
        [
            create_dex_df(Y_train, Y_train_pred, 'Train', paths_train, min_ship_score),
            create_dex_df(Y_valid, Y_valid_pred, 'Validation', paths_valid, min_ship_score),
            create_dex_df(Y_test, Y_test_pred, 'Test', paths_test, min_ship_score)
        ]
    )
    print(df_dex)
    df_dex.to_csv(os.path.join(args['model_dir'], 'dex.csv'), index = False)


