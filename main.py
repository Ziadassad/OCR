import os
import numpy as np
import cv2
import tensorflow as tf
from load_training_data import *
from training_data import *
from process import *
from histogram_word_detection import *
from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# a = load_training_data()
# a.load_image()
#
# training_data()

# model = tf.keras.models.load_model('cnn.model')


CATEGORIES = ["ز", "د", "م", "ک", "ر", "و"]


image = cv2.imread('train-data\\Z\\z4.png',  cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('train-data\\Y\\y3.png',  cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('train-data\\K\\k_67.png',  cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('train-data\\D\\d40.png',  cv2.IMREAD_GRAYSCALE)

image5 = cv2.imread('train-data\\test\\test4.png',  cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread('train-data\\test\\kurdistan.png',  cv2.IMREAD_GRAYSCALE)


ret, thresh = cv2.threshold(image5, 0, 255, cv2.THRESH_BINARY_INV)


im = image5

im = cv2.resize(im, (500, 500))

cv2.imshow("th", thresh)

detect_word = histogram_word_detection(thresh)
horizontal = detect_word.Horizontal_histogram(thresh)
point, imageV = detect_word.Vertical_histogram(horizontal)

word_images = detect_word.getImageOfWords(point, imageV)

for im in reversed(word_images):
    # print("i")
    p = process(im)
    p.get_letter()

# p = process(word_images[0])
# p.get_letter()



# def preper(image, i):
#     # cv2.imshow('e', image)
#     # ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
#     im = np.array(image)
#     new = cv2.resize(im, (28, 28))
#     cv2.imshow(str(i), new)
#     return new.reshape(-1, 28, 28, 1)
#
#
# prediction = model.predict([preper(image5, 1)])
# print(np.argmax(prediction[0]))
# print(prediction)
# print(max(prediction[0]))
# print(CATEGORIES[int(np.argmax(prediction[0]))])


# plt.imshow(mask, cmap=plt.gray())
# plt.show()



# cv2.imshow('image', word_images[0])
cv2.waitKey(0)





# CATEGORIES = [ "D", "M", "A",  "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# dir = 'D:\\OCR\\train-data'
#
#
# for category in CATEGORIES:
#     print(category)
#     i = 0
#     path = os.path.join(dir, category)
#     for img in os.listdir(path):
#         gray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
#         cv2.imwrite(f"D:\\OCR\\train-data\\{category}1\\{category+str(i)}.png", thresh)
#         i += 1