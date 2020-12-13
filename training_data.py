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
        CATEGORIES = ["A", "D", "KL", "R", "W", "N", "TL", "SL"]

        pickle_in = open("x.pickle", "rb")
        train_images = pickle.load(pickle_in)

        pickle_in = open("y.pickle", "rb")
        train_labels = pickle.load(pickle_in)

        train_images = np.array(train_images).reshape(-1, 28, 28, 1)
        train_labels = np.array(train_labels)

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
                tf.keras.layers.Conv2D(60, (5, 5), activation="relu", input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(60, (5, 5), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(60//2, (3, 3), activation="relu"),
                tf.keras.layers.Conv2D(60//2, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(500, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(8, activation="softmax")
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
            epochs=10,
            verbose=1,
            validation_data=(train_images, train_labels),
            shuffle=1,
            callbacks=[],
        )
        stop = time.time()
        print("Training time:", stop - start, "seconds")

        print("train score:", model.evaluate(train_images, train_labels, batch_size=128))
        # print("test score:", model.evaluate(test_images, test_labels, batch_size=128))

        model.save('cnn.model')