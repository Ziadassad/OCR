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

CATEGORIES = ["A", "D", "KL", "R", "W", "N", "TL", "SL"]
# CATEGORIES = ["ا", "د", "ک", "ر", "و", "ن", "ت"]


image = cv2.imread('train-data\\Z\\z4.png',  cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('train-data\\Y\\y3.png',  cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('train-data\\D\\_67.png',  cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('train-data\\Ztest\\C1.PNG',  cv2.IMREAD_GRAYSCALE)

image5 = cv2.imread('train-data\\Ztest\\test4.png',  cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread('train-data\\Ztest\\kurdistan.png',  cv2.IMREAD_GRAYSCALE)

# re, t = cv2.threshold(image6, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite("C:\\Users\\ZiadPro\\Desktop\\pycharm\\OCR\\train-data\\test\\kurdistan2.png", t)
# tr

ret, thresh = cv2.threshold(image6, 127, 255, cv2.THRESH_BINARY_INV)


im = thresh

im = cv2.resize(im, (500, 500))

cv2.imshow("th", im)

detect_word = histogram_word_detection(im, "word")
horizontal = detect_word.Horizontal_histogram(thresh)

point, imageV = detect_word.Vertical_histogram(horizontal[0])
word_images = detect_word.getImageOfWords(point, imageV)

# cv2.imshow("rr2", horizontal[0])
# cv2.imshow("rrr", word_images[0])

for im in reversed(word_images):
    p = process(im)
    # print(im.shape)
    p.get_letter()

# p = process(word_images[0])
# p.get_letter()


# def nothing(x):
#     pass
#
# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
#
# model = tf.keras.models.load_model('cnn.model')
# def prepro(img):
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img/255
#     return img
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     _, frame = cap.read()
#     frame = np.array(frame)
#     # cv2.imshow("org", frame)
#     # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray", img)
#     # ret, img = cv2.threshold(img, 120, 205, cv2.THRESH_BINARY_INV)
#     # cv2.imshow("thrsh", img)
#     # img = cv2.resize(img, (28, 28))
#     # img = prepro(img)
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     l_h = cv2.getTrackbarPos("LH", "Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")
#
#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])
#
#     mask = cv2.inRange(hsv, l_b, u_b)
#
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("res", res)
#
#     img = cv2.resize(mask, (28, 28))
#     img = prepro(img)
#     img = img.reshape(-1, 28, 28, 1)
#     # clas = int(model.predict_classes(img))
#     predict = model.predict(img)
#     val = np.amax(predict)
#     let = CATEGORIES[int(np.argmax(predict[0]))]
#     print(let, ' ', val)
#     # print(clas, ' ', val)
#
#     cv2.putText(frame, let + "  %" + str(val), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#     cv2.imshow("org", frame)
#     cv2.waitKey(1)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




# cv2.imshow('image', word_images[0])
cv2.waitKey(0)