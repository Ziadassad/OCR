import os
import cv2
import pickle
import numpy as np

class load_training_data:
    def __init__(self):
        self.list = []
        self.DATADIR = 'D:\\OCR\\train-data'
        # self.CATEGORIES = ["Z", "Y", "M", "S", "D", "A", "Aa", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.CATEGORIES = ["Z", "D", "M"]

    def load_image(self):
        i = 0
        list_data = self.list.copy()
        dir = self.DATADIR
        catog = self.CATEGORIES
        for category in catog:
            print(category)
            class_name = catog.index(category)
            print(class_name)
            path = os.path.join(dir, category)
            for img in os.listdir(path):
                gray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_image = cv2.resize(gray, (28, 28))
                list_data.append([np.array(new_image), np.array(class_name)])

        x_image = []
        y_label = []
        for x, y in list_data:
            x_image.append(x)
            y_label.append(y)

        x_image = np.array(x_image).reshape(-1, 28, 28, 1)
        pickle_out = open('x.pickle', 'wb')
        pickle.dump(x_image, pickle_out)
        pickle_out.close()
        pickle_out2 = open('y.pickle', 'wb')
        pickle.dump(y_label, pickle_out2)
        pickle_out.close()
        print("complet")
        return list_data