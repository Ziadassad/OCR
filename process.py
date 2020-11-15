import cv2
from load_training_data import *
from training_data import *
from histogram_word_detection import *

class process:
    def __init__(self, image):
        self.image = image
        self.words = []
        # cv2.imshow('d', image)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
        detect_word = histogram_word_detection(thresh)
        point, images = detect_word.Vertical_histogram(thresh)

        for p1, p2 in point:
            # print(p1)
            self.words.append(image[:, p1:p2])
            # cv2.imshow(str(p1), image[:, p1:p2])


    def get_letter(self):
        CATEGORIES = ["ز", "د", "م", "ک", "ر", "و", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
        word = []
        print(len(self.words))
        i = 0
        for im in self.words:
            im = 255 - im
            # cv2.imshow(str(i), im)
            # print(im)
            detect_word = histogram_word_detection(im)
            im = detect_word.Horizontal_histogram(im)
            word.append(im)
            # cv2.imshow(str(i), im)
            # i += 1

        def preper(image, i):
            # cv2.imshow('e', image)
            ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
            im = np.array(thresh)
            new = cv2.resize(im, (28, 28))
            cv2.imshow(str(i), new)
            return new.reshape(-1, 28, 28, 1)

        model = tf.keras.models.load_model('cnn.model')

        print(len(word))
        letter = ""
        for w in word:
            p = preper(w, i)
            prediction = model.predict([p])
            # print(np.argmax(prediction[0]))
            # print(prediction)
            # print(max(prediction[0]))
            # print(CATEGORIES[int(np.argmax(prediction[0]))])
            letter = letter + CATEGORIES[int(np.argmax(prediction[0]))]
            i += 1

        print(letter[::-1])
        # cv2.imshow('e', p)
        # prediction = model.predict([preper(word[4], 1)])
        # print(np.argmax(prediction[0]))
        # print(prediction)
        # print(max(prediction[0]))
        # print(CATEGORIES[int(np.argmax(prediction[0]))])
        # self.letter = CATEGORIES[int(np.argmax(prediction[0]))]
        # return self.letter