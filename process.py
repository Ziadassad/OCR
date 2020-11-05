import numpy as np
from load_training_data import *
from training_data import *

class process:
    def __init__(self, image):
        CATEGORIES = ["ز", "د", "م", "ی", "س", "د", "ا", "آ", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]

        def preper(image):
            im = np.array(image)
            new = cv2.resize(im, (28, 28))
            return new.reshape(-1, 28, 28, 1)

        # training_data()
        model = tf.keras.models.load_model('cnn.model')

        prediction = model.predict([preper(image)])
        print(np.argmax(prediction[0][0]))
        print(prediction)
        print(max(prediction[0]))
        self.letter = CATEGORIES[int(np.argmax(prediction[0]))]

    def get_letter(self):
        return self.letter,