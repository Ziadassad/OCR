import os
import numpy as np
import cv2
import tensorflow as tf
from load_training_data import *
from training_data import *
from process import *
from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw



# a = load_training_data()
# a.load_image()
#
# training_data()

model = tf.keras.models.load_model('cnn.model')

# window = Tk()
# window.title("OCR Kurdish")
# window.geometry("800x800")
#
# lbl = Label(window, text="Write digits with your mouse in the gray square", font=('Arial Blod', 15))
# lbl.grid(column=5, row=0)
#
# white = (255, 255, 255)
# black = (0, 0, 0)
#
# canvas_x = 200
# canvas_y = 200
#
# image = Image.new("RGB", (canvas_x, canvas_y), white)
# draw = ImageDraw.Draw(image)
# xpoints = []
# ypoints = []
# x2points = []
# y2points = []
#
# w = Canvas(window,
#                width=canvas_x,
#                height=canvas_y, bg='gray85')
# w.grid(column=5, row=2)
#
#
# def paint(event):
#     x1, y1 = (event.x - 4), (event.y - 4)
#     x2, y2 = (event.x + 4), (event.y + 4)
#     w.create_oval(x1, y1, x2, y2, fill='black')
#     xpoints.append(x1)
#     ypoints.append(y1)
#     x2points.append(x2)
#     y2points.append(y2)
#
# def imagen():
#     global counter
#     global xpoints
#     global ypoints
#     global x2points
#     global y2points
#     image1 = Image.new("RGB", (canvas_x, canvas_y), black)
#     draw = ImageDraw.Draw(image1)
#
#     elementos = len(xpoints)
#
#     for p in range(elementos):
#         x = xpoints[p]
#         y = ypoints[p]
#         x2 = x2points[p]
#         y2 = y2points[p]
#         draw.ellipse((x, y, x2, y2), 'white')
#         w.create_oval(x - 4, y - 4, x2 + 4, y2 + 4, outline='gray85', fill='gray85')
#
#
#     result = process(image1)
#     letter = result.get_letter()
#     print(letter)
#     lbl2 = Label(window, text=letter, font=('Arial Bold', 20))
#     lbl2.grid(column=5, row=10)
#
#     xpoints = []
#     ypoints = []
#     x2points = []
#     y2points = []
#
# w = Canvas(window,
#             width=canvas_x,
#             height=canvas_y, bg='gray85')
# w.grid(column=5, row=2)
#
# w1 = Canvas(window, width=200, height=200, bg='gray95')
# w1.grid(column=3, row=10)
#
# w.bind("<B1-Motion>", paint)
# button = tk.Button(window, text='Save image', width=25, command=imagen)
# button.grid(column=5, row=4)
#
#
# window.mainloop()

# CATEGORIES = ["ز", "ی", "م", "س", "د", "ا", "آ", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
# CATEGORIES = ["Z", "Y", "S", "D", "A", "Aa", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CATEGORIES = ["ز", "د", "م"]


image = cv2.imread('D:\\OCR\\train-data\\Z\\z4.png',  cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('D:\\OCR\\train-data\\Y\\y3.png',  cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('D:\\OCR\\train-data\\M\\m27.png',  cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('D:\\OCR\\train-data\\D\\d13.png',  cv2.IMREAD_GRAYSCALE)

image5 = cv2.imread('D:\\OCR\\train-data\\test\\test4.png',  cv2.IMREAD_GRAYSCALE)

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape and rescale data for the CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images / 255, test_images / 255



pro = process(image)

print(pro.get_letter())





# cv2.imshow('image', thresh)

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