
import numpy as np
import cv2
import tensorflow as tf
from load_training_data import *
from training_data import *
from process import *
from histogram_word_detection import *
from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from matplotlib import pyplot as plt



# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# a = load_training_data()
# a.load_image()
#
# training_data()


CATEGORIES = ["ا", "ئ", "ە", "ب", "د", "ك", "ر", "ڕ", "و", "وو", "ن", "ت", "چ", "ف", "گ",
              "ه", "ج", "ل", "ڵ", "م", "ۆ", "پ", "ق", "س", "ش", "ح", "ع", "ڤ", "خ", "غ", "ی", "ێ", "ز", "ژ", "کو", "ستا"]


image = cv2.imread('train-data\\Z\\z4.png',  cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('train-data\\Y\\y3.png',  cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('train-data\\D\\_67.png',  cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('train-data\\Ztest\\C1.PNG',  cv2.IMREAD_GRAYSCALE)

image5 = cv2.imread('train-data\\Ztest\\test4.png',  cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread('train-data\\Ztest\\kurdistan.png',  cv2.IMREAD_GRAYSCALE)

re, t = cv2.threshold(image5, 127, 255, cv2.THRESH_BINARY)
# cv2.imwrite("C:\\Users\\ZiadPro\\Desktop\\pycharm\\OCR\\train-data\\test\\kurdistan2.png", t)
# tr

ret, thresh = cv2.threshold(image5, 127, 255, cv2.THRESH_BINARY_INV)


im = thresh

im = cv2.resize(im, (500, 500))
t = cv2.resize(t, (500, 500))


# detect_word = histogram_word_detection(im, "word")
# horizontal = detect_word.Horizontal_histogram(thresh)
# point, imageV = detect_word.Vertical_histogram(horizontal[0])
# word_images = detect_word.getImageOfWords(point, imageV)

# cv2.imshow("rr2", horizontal[0])
# cv2.imshow("rrr", word_images[0])

# print(len(word_images))

# i = 0
# for im in reversed(word_images[0]):
#     # im = cv2.resize(im, (500, 500))
#     p = process(im)
#     # print(im.shape)
#     # cv2.imshow(str(i), im)
#     p.get_letter()
#     # i += 1

# p = process(image)
# p.get_letter()




model = tf.keras.models.load_model('cnn.model')
def prepro(img):
    img = cv2.equalizeHist(img)
    img = img/255
    return img

cap = cv2.VideoCapture(1)

root = tk.Tk()

rgb = tk.Label(root)
rgb.grid(column=1, row=0)
image = tk.Label(root)
image.grid(column=1, row=1)
thresh_hold = tk.Label(root)
thresh_hold.grid(column=2, row=0)

label_one = tk.Label(root)
label_one.grid(column=2, row=1)

label_one.config(font=("Courier", 44))

imgn = np.zeros((300, 300))
img = Image.fromarray(imgn)
img = ImageTk.PhotoImage(image=img)
image.img = img
image.configure(image=img)

imthresh = imgn

def proces():
    _, frame = cap.read()
    imrgb = np.array(frame)
    gray = cv2.cvtColor(imrgb, cv2.COLOR_RGB2GRAY)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # imthresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    ret, imthresh1 = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, imthresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # imthresh = cv2.dilate(thresh, None)

    bit = cv2.bitwise_or(imthresh, imthresh1)

    erode = cv2.erode(bit, None, iterations=1)

    erode = erode[20: 450, 50:620]

    imthresh = cv2.resize(erode, (500, 500))
    img = Image.fromarray(imthresh)
    img = ImageTk.PhotoImage(image=img)
    image.img = img
    image.configure(image=img)

    ret, imthresh = cv2.threshold(imthresh, 0, 255, cv2.THRESH_BINARY_INV)

    detect_word = histogram_word_detection(imthresh, "word")
    horizontal = detect_word.Horizontal_histogram(imthresh)
    point, imageV = detect_word.Vertical_histogram(horizontal[0])
    word_images = detect_word.getImageOfWords(point, imageV)

    # i = 0
    # for im in reversed(word_images[0]):
    #     im = cv2.resize(im, (500, 500))
    #     p = process(im)
    #     print(im.shape)
    #     # cv2.imshow(str(i), im)
    #     p.get_letter()
    #     i += 1

    p = process(word_images[0])
    result = p.get_letter()

    label_one.config(text=result)

def video_stream():
    _, frame = cap.read()
    imrgb = np.array(frame)
    imrgb = cv2.cvtColor(imrgb, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imrgb, cv2.COLOR_RGB2GRAY)
    # print(imrgb.shape)
    ret, imthresh1 = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

    # imrgb = cv2.GaussianBlur(imrgb, (7, 7), 0)
    # hsv = cv2.cvtColor(imrgb, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # imthreshAd = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # imthresh = imthresh[120: 350, 190:450]
    # imthresh = cv2.resize(imthresh, (500, 500))

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, imthresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # imthresh = cv2.dilate(imthresh, None)
    bit = cv2.bitwise_or(imthresh, imthresh1)

    erode = cv2.erode(bit, None, iterations=1)

    erode = erode[20: 450, 50:620]


    img = Image.fromarray(imrgb)
    imgtkrgb = ImageTk.PhotoImage(image=img)

    imgt = Image.fromarray(erode)
    imgtkth = ImageTk.PhotoImage(image=imgt)

    rgb.imgtkgray = imgtkrgb
    rgb.configure(image=imgtkrgb)

    thresh_hold.imgtkth = imgtkth
    thresh_hold.configure(image=imgtkth)

    root.after(10, video_stream)


video_stream()

button = tk.Button(root, text='Save image', width=20, command=proces).grid(column=1, row=2)
tk.mainloop()


# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# while True:
#     _, frame = cap.read()
#     frame = np.array(frame)
#     cv2.imshow("org", frame)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # print(img.shape)
#     cv2.imshow("gray", img)
#     ret, img = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow("thrsh", img)
#     img = cv2.resize(img, (28, 28))
#     img = prepro(img)
#
#     # img = cv2.medianBlur(img, 5)
#
#     # ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     # th2 = cv2.adaptiveThreshold(img, 205, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 2)
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
#     mask = img[70:400, 70:560]
#
#     # res = cv2.bitwise_and(frame, frame, mask=mask)
#     # cv2.imshow("res", res)
#
#     # cv2.imshow("frame", frame)
#     # cv2.imshow("mask", mask)
#
#     # img = cv2.resize(mask, (28, 28))
#     # img = prepro(img)
#     # img = img.reshape(-1, 28, 28, 1)
#     # predict = model.predict(img)
#     # val = np.amax(predict)
#     # let = CATEGORIES[int(np.argmax(predict[0]))]
#     # print(let, ' ', val)
#     # # print(clas, ' ', val)
#     #
#     # cv2.putText(frame, let + "  %" + str(val), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#     # cv2.imshow("org", frame)
#     cv2.waitKey(1)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




# cv2.imshow('image', word_images[0])
cv2.waitKey(0)