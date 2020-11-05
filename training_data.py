# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
# from keras.utils.np_utils import to_categorical
# from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from numpy import argmax
from tkinter import *
import tkinter as tk
import pickle
import time
from PIL import Image, ImageDraw

class training_data:
    def __init__(self):
        # CATEGORIES = ["Z", "Y", "S", "D", "A", "Aa", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        CATEGORIES = ["Z", "D"]
        pickle_in = open("x.pickle", "rb")
        train_images = pickle.load(pickle_in)

        pickle_in = open("y.pickle", "rb")
        train_labels = pickle.load(pickle_in)

        train_images = np.array(train_images).reshape(-1, 28, 28, 1)
        # x = x.reshape(192, 28, 28, 1)
        train_labels = np.array(train_labels)

        # y = to_categorical(y)

        train_images = train_images / 255

        # mnist = tf.keras.datasets.mnist
        # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        #
        # # reshape and rescale data for the CNN
        # train_images = train_images.reshape(60000, 28, 28, 1)
        # test_images = test_images.reshape(10000, 28, 28, 1)
        # train_images, test_images = train_images / 255, test_images / 255

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        print(f"Model parameters: {model.count_params():,}")

        start = time.time()

        model.fit(
            train_images,
            train_labels,
            batch_size=128,
            epochs=1,
            verbose=1,
            # validation_data=(t, test_labels),
            callbacks=[],
        )
        stop = time.time()
        print("Training time:", stop - start, "seconds")

        print("train score:", model.evaluate(train_images, train_labels, batch_size=128))
        # print("test score:", model.evaluate(test_images, test_labels, batch_size=128))

        model.save('cnn.model')

        # model = Sequential()
        #
        # model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        #
        # model.add(Dense(64))
        #
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))
        #
        # print(len(CATEGORIES))
        #
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])
        #
        # model.fit(x, y, batch_size=32, epochs=2, validation_split=0.9)
        #
        # model.save('cnn.model')

        # model.compile(
        #     optimizer="rmsprop",
        #     loss="sparse_categorical_crossentropy")
        # model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.save('cnn.model2')