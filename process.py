import cv2
from load_training_data import *
from training_data import *
from histogram_word_detection import *
from PIL import Image, ImageFilter


class process:

    def __init__(self, image):
        # "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"
        # ["A", "AL", "AC", "B", "D", "KL", "R", "RR", "W", "WW", "N", "TS", "TL", "SL", "CH", "F", "G",
        #  "H", "C", "L", "LL", "M", "O", "P", "Q", "S", "SL", "sh", "U", "UU", "V", "X", "XX", "Y", "YY",
        #  "Z", "ZH"]
        self.CATEGORIES = ["ا", "ئ", "ە", "ب", "د", "ك", "ر", "ڕ", "و", "وو", "ن", "ت", "س", "چ", "ف", "گ",
                           "ه", "ج", "ل", "ڵ" "م", "ۆ", "پ", "ق", "س", "ش", "ح", "ع", "ڤ", "خ", "غ", "ی", "ێ", "ز", "ژ"]
        # self.CATEGORIES = ["ا", "د", "ک", "ر", "و", "ن", "ت", "س"]
        self.image = image
        self.words = []
        # cv2.imshow('d', image)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
        self.detect_word = histogram_word_detection(thresh, "letter")
        point, images = self.detect_word.Vertical_histogram(thresh)

        for p1, p2 in point:
            # print(p1)
            self.words.append(image[:, p1:p2])
            # cv2.imshow(str(p1), image[:, p1:p2])

        self.word = []
        i = 0
        for im in self.words:
            detect_word = histogram_word_detection(im, "letter")
            im = detect_word.Horizontal_histogram(im)
            self.word.append(im)
            # cv2.imshow(str(i), im)
            # i += 1


    # model = tf.keras.models.load_model('cnn.model')

    def predict(self, p):
        model = tf.keras.models.load_model('cnn.model')
        prediction = model.predict([p])
        letter = self.CATEGORIES[int(np.argmax(prediction[0]))]
        print(letter)
        return letter


    #
    #


    def scan(self, image):
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
        detect_word = histogram_word_detection(thresh, "letter")
        point, image = detect_word.Vertical_histogram(thresh)

        print(len(image))
        images = []
        for p1, p2 in point:
            images.append(image[:, p1:p2])
            # cv2.imshow(str(p1), )
        # cv2.imshow(str(p1), images[0])

    def get_letter(self):

        def preper(image, i):
            cv2.imshow(str(i), image)
            image = np.array(image)
            new = cv2.resize(image, (28, 28))
            cv2.imshow(str(i), image)
            return new.reshape(-1, 28, 28, 1)

        # self.findLettercon()
        letter = ""
        i = 0
        for w in self.word:
            ret, image = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY)
            print(w.shape)
            # cv2.imshow(str(i), image)
            # print(w.shape)

            p = preper(image, i)
            letter = letter + self.predict(p)
            i += 1

        print(letter[::-1], end=" ")