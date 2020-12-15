import cv2
from load_training_data import *
from training_data import *
from histogram_word_detection import *


class process:

    def __init__(self, image):
        self.CATEGORIES = ["ا", "د", "ک", "ر", "و", "ن", "ت", "س", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
        self.image = image
        self.words = []
        cv2.imshow('d', image)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
        detect_word = histogram_word_detection(thresh, "letter")
        point, images = detect_word.Vertical_histogram(thresh)


        for p1, p2 in point:
            # print(p1)
            self.words.append(image[:, p1:p2])
            # cv2.imshow(str(p1), image[:, p1:p2])

    model = tf.keras.models.load_model('cnn.model')

    def predict(self, p):
        model = tf.keras.models.load_model('cnn.model')
        prediction = model.predict([p])
        # print(np.argmax(prediction[0]))
        # print(prediction)
        print(max(prediction[0]))
        # print(CATEGORIES[int(np.argmax(prediction[0]))])
        letter = self.CATEGORIES[int(np.argmax(prediction[0]))]
        # print(letter)
        return letter

    def get_letter(self):
        word = []

        # print(len(self.words))
        i = 0
        for im in self.words:
            detect_word = histogram_word_detection(im, "letter")
            im = detect_word.Horizontal_histogram(im)
            word.append(im)
            # cv2.imshow(str(i), im)
            # i += 1

        def preper(image, i):
            cv2.imshow(str(i), image)
            image = np.array(image)
            new = cv2.resize(image, (28, 28))
            cv2.imshow(str(i), image)
            # return new
            return new.reshape(-1, 28, 28, 1)

        # print(len(word))
        letter = ""
        i = 0
        for w in word:
            ret, image = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY)
            # cv2.imshow(str(i), image)
            # print(w.shape)
            # if w.shape[1] > 60:
            #     pass
            #     # self.scan(image)
            # else:
            p = preper(image, i)
            letter = letter + self.predict(p)
            i += 1

        print(letter[::-1], end=" ")