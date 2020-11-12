import cv2
from load_training_data import *
from training_data import *
from histogram_word_detection import *

class process:
    def __init__(self, image):
        CATEGORIES = ["ز", "د", "م", "ی", "س", "د", "ا", "آ", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
        im = image[:, 150:190]
        im = 255 - im
        detect_word = histogram_word_detection(im)
        im = detect_word.Horizontal_histogram(im)
        cv2.imshow('b', im)
        def preper(image):
            # cv2.imshow('e', image)
            ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
            im = np.array(thresh)
            new = cv2.resize(im, (28, 28))
            cv2.imshow('e', new)
            return new.reshape(-1, 28, 28, 1)

        # training_data()
        model = tf.keras.models.load_model('cnn.model')
        # p = preper(im)
        # cv2.imshow('e', p)
        prediction = model.predict([preper(im)])
        print(np.argmax(prediction[0]))
        print(prediction)
        print(max(prediction[0]))
        print(CATEGORIES[int(np.argmax(prediction[0]))])
        self.letter = CATEGORIES[int(np.argmax(prediction[0]))]

    def get_letter(self):
        return self.letter