import numpy as np
import cv2

class histogram_word_detection:

    def __init__(self, orginal_image, flag):
        self.h1 = 0
        self.h2 = 0
        self.flag = flag
        if flag == "word":
            self.orginal_image = cv2.resize(orginal_image, (500, 500))
        self.imageOrg = orginal_image
        # bo wargrtne point wordakan
        self.point_image = []

    def Horizontal_histogram(self, image):

        if self.flag == "word":
            image = cv2.resize(image, (500, 500))
        else:
            image = cv2.resize(image, (500, image.shape[0]))
            # print(image.shape[0])
        im = image
        im = 255 - im

        # Calculate horizontal projection
        proj = np.sum(im, 1)
        m = np.max(proj)

        horizontal = np.zeros((image.shape[0], image.shape[1]))

        h, w = horizontal.shape

        # Draw a line for each row
        for row in range(im.shape[0]):
           cv2.line(horizontal, (0, row), (int(proj[row]*w/m), row), (255, 255, 255), 1)

        rowSentence = []
        store = []

        # bo dyare krdne hamw row yakan
        if self.flag != "word":
            for i in range(0, h):
                if horizontal[i, 20] > 200:
                    # print(i)
                    store.append(i)
            # cv2.imshow("letter", self.imageOrg[min(store):max(store), :])
            return self.imageOrg[min(store):max(store), :]


        for i in range(0, h):
            # print(i)
            if horizontal[i, 20] > 220:
                store.append(i)
                if horizontal[i+1, 20] < 20:
                    rowSentence.append([min(store), max(store)])
                    store.clear()

        print(rowSentence)
        c = 0

        # bo dyare krdne sentence
        sentence = []

        for h1, h2 in rowSentence:
            if len(rowSentence)-1 != c:
                start = rowSentence[c + 1][0]
                end = rowSentence[c + 1][1]
                distance = start - h2

                if distance < 30:
                    print(distance)
                    sentence.append([h1, end])
                else:
                    if c == 0:
                        sentence.append([h1, h2])
                    else:
                        end = max(sentence)
                        if end[1] < h1:
                            sentence.append([h1, h2])
                c += 1
            else:
                if c == 0:
                    sentence.append([h1, h2])
                else:
                    end = max(sentence)
                    if end[1] < h1:
                        sentence.append([h1, h2])

        print(sentence)

        imsent = []  # bo save krdne aw sentence nay boman darchwa
        for h1, h2 in sentence:
            # cv2.line(image, (0, h1), (w, h1), color=(0, 255, 255), thickness=2)
            # cv2.line(image, (0, h2), (w, h2), color=(0, 255, 255), thickness=2)
            imsent.append(self.orginal_image[h1:h2, :])

        cv2.imshow('result', horizontal)
        # self.h1, self.h2 = min(store), max(store)
        cv2.imshow('re', image)
        return imsent

    def Vertical_histogram(self, image):
        # im1 = image[self.h1: self.h2, :]

        image = 255 - image
        proj = np.sum(image, 0)

        m = np.max(proj)
        vertical = np.zeros((500, 500))

        h, w = vertical.shape

        # # Draw a line for each row
        for row in range(image.shape[1]):
           cv2.line(vertical, (row, 0), (row, int(proj[row]*w/m)), (255, 255, 255), 1)

        store = []

        point = []
        for i in range(0, w):
            if vertical[60, i] > 200:
                store.append(i)
                if vertical[60, i+1] < 10:
                    point.append([min(store), max(store)])
                    store.clear()

        # for img in point:
        #     cv2.line(image, (min(img), 0), (min(img), h), color=(255, 255, 255), thickness=1)
        #     cv2.line(image, (max(img), 0), (max(img), h), color=(255, 255, 255), thickness=1)

        cv2.imshow('vertical', vertical)
        # cv2.imshow('vertical', image)

        return point, image



    def getImageOfWords(self, point_image, image):
        point = point_image
        images = []

        # print(point)
        distance = 0
        newPoint = []
        check = []
        cp = 0
        cn = 0
        if len(point) > 1:
            for w1, w2 in point:
                if cp != len(point) - 1:
                    start = point[cp + 1][0]
                    # end = point[cp+1][1]
                    distance = start - w2
                    if distance < 22:
                        check.append([w1, w2])
                    else:
                        if check == []:
                            newPoint.append([w1, w2])
                            cn += 1
                        else:
                            check.append([w1, w2])
                            startN = check[0][0]
                            endN = check[-1][1]
                            newPoint.append([startN, endN])
                            cn += 1
                            check.clear()
                    cp += 1
                    # print(distance)
                    # print(newPoint)
                else:
                    if cn > 0:
                        cn -= 1
                        if newPoint[cn][1] != w2 and check == []:
                            newPoint.append([w1, w2])
                            cn += 1
                            cp += 1
                        else:
                            newPoint.append([check[0][0], w2])
                            cn += 1
                    else:
                        newPoint.append([check[0][0], w2])
                        cn += 1
        else:
            newPoint = point



        # print(newPoint)
        self.point_image = newPoint

        for w1, w2 in self.point_image:
            # print(w1, ' ', w2)
            img = image[:, w1: w2]
            images.append(img)

        return images