import numpy as np
import cv2

class histogram_word_detection:

    def __init__(self, orginal_image):
        self.h1 = 0
        self.h2 = 0
        image = cv2.resize(orginal_image, (500, 500))
        self.imageOrg = image
        # bo wargrtne point wordakan
        self.point_image = []

    def Horizontal_histogram(self, image):

        image = cv2.resize(image, (500, 500))
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

        s = []

        for i in range(0, h):
            if horizontal[i, 20] > 200:
                # print(i)
                s.append(i)

        # print(min(s), '  ', max(s))

        cv2.line(im, (0, min(s)), (w, min(s)), color=(0, 0, 255), thickness=2)
        cv2.line(im, (0, max(s)), (w, max(s)), color=(0, 0, 255), thickness=2)
        cv2.imshow('result', horizontal)
        self.h1, self.h2 = min(s), max(s)
        # cv2.imshow('re', image[self.h1: self.h2])
        return self.imageOrg[self.h1: self.h2]

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
            if vertical[10, i] > 200:
                store.append(i)
                if vertical[10, i+1] < 10:
                    point.append([min(store), max(store)])
                    store.clear()

        # for img in point:
        #     cv2.line(image, (min(img), 0), (min(img), h), color=(255, 255, 255), thickness=1)
        #     cv2.line(image, (max(img), 0), (max(img), h), color=(255, 255, 255), thickness=1)

        cv2.imshow('vertical', vertical)
        # cv2.imshow('vertical', image)

        return point, image


        # print(point)
        # distance = 0
        # newPoint = []
        # check = []
        # cp = 0
        # cn = 0
        # if len(point) > 1:
        #     for w1, w2 in point:
        #         if cp != len(point)-1:
        #             start = point[cp+1][0]
        #             # end = point[cp+1][1]
        #             distance = start - w2
        #             if distance < 22:
        #                 check.append([w1, w2])
        #             else:
        #                 if check == []:
        #                     newPoint.append([w1, w2])
        #                     cn += 1
        #                 else:
        #                     check.append([w1, w2])
        #                     startN = check[0][0]
        #                     endN = check[-1][1]
        #                     newPoint.append([startN, endN])
        #                     cn += 1
        #                     check.clear()
        #             cp += 1
        #             # print(distance)
        #             # print(newPoint)
        #         else:
        #             if cn > 0:
        #                 cn -= 1
        #                 if newPoint[cn][1] != w2 and check == []:
        #                     newPoint.append([w1, w2])
        #                     cn += 1
        #                     cp += 1
        #                 else:
        #                     newPoint.append([check[0][0], w2])
        #                     cn += 1
        #             else:
        #                 newPoint.append([check[0][0], w2])
        #                 cn += 1

                    # print(w1, ' ', w2)
                    # start = point[cp+1][0]
                    # end = point[cp+1][1]
                    # distance = start - w2
                    # if distance < 22:
                    #     if cn > 0:
                    #         startN = newPoint[cn][0]
                    #         endN = newPoint[cn][1]
                    #         print('d', distance)
                    #         print(endN)
                    #         d = endN - w1
                    #         print(d)
                    #         if d > 1:
                    #             del newPoint[cn]
                    #             newPoint.append([startN, end])
                    #             # cn -= 1
                    #         else:
                    #             newPoint.append([w1, w2])
                    #             cn += 1
                    #     else:
                    #         newPoint.append([w1, end])
                    #         cn += 1
                    # else:
                    #     if cn > 0:
                    #         endn = newPoint[cn - 1][1]
                    #         startn = newPoint[cn][1]
                    #         # print('n', startn)
                    #         # print('t', w1)
                    #         if endn != w2:
                    #             if startn > w1:
                    #                 # print("yes")
                    #                 pass
                    #             else:
                    #                 # print("yes")
                    #                 newPoint.append([w1, w2])
                    #                 cn += 1
                            # if startn < w1:
                            #     print("yes")
                            #     newPoint.append([w1, w2])
                            #     cn += 1
                            # else:
                            #     newPoint.append([w1, w2])
                            #     cn += 1

                        # else:
                        #     newPoint.append([w1, w2])
                        #     cn+1
                    # print(distance)
                # else:
                #     if newPoint[cn][1] != w2 and check == []:
                #         newPoint.append([w1, w2])
                #         cp += 1
                #     else:
                #         newPoint.append([check[0][0], w2])
                # c = 0
                # for w1, w2 in newPoint:
                #     # print(w1, ' ', w2)
                #     if c != len(newPoint)-1:
                #         start = newPoint[c + 1][0]
                #         end = newPoint[c+1][1]
                #         if start < w2:
                #             del newPoint[c]
                #     else:
                #         print(len(newPoint))
                #         start = newPoint[c - 1][0]
                #         end = newPoint[c - 1][1]
                #         if end > w1:
                #             del newPoint[c]
                #             del newPoint[c-1]
                #             newPoint.append([start, w2])
                #         print(w1)
                #     # print(end)
                #     c += 1
        # else:
        #     newPoint = point
        #
        # print(newPoint)
        # self.point_image = newPoint
        # for img in newPoint:
        #     cv2.line(image, (min(img), 0), (min(img), h), color=(255, 255, 255), thickness=1)
        #     cv2.line(image, (max(img), 0), (max(img), h), color=(255, 255, 255), thickness=1)


        # word = self.image[self.h1: self.h2, w1: w2]
        # cv2.imshow('res', vertical)
        # cv2.imshow('vertical', image)
        # cv2.imshow('word', word)
        # return image


    def getImageOfWords(self, point_image, image):
        point = point_image
        images = []

        print(point)
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



        print(newPoint)
        self.point_image = newPoint

        for w1, w2 in self.point_image:
            # print(w1, ' ', w2)
            img = image[:, w1: w2]
            images.append(img)

        return images


    # if i != len(point) - 1:
    #     distance = point[i + 1][0] - point[i][1]
    #     start = point[i][0]
    #     end = point[i + 1][1]
    #     if distance < 25:
    #         if c > 0:
    #             pre = newPoint[c - 1][1]
    #             # print(pre, ' ', start)
    #             if pre > start:
    #                 # del newPoint[c-1]
    #                 newPoint.append([point[i - 1][0], end])
    #             else:
    #                 newPoint.append([start, end])
    #         else:
    #             newPoint.append([start, end])
    #         c += 1
    #     else:
    #         if c > 0:
    #             pre = newPoint[c - 1][1]
    #             # print(pre, ' ', start)
    #             if pre > start:
    #                 continue
    #             else:
    #                 newPoint.append([start, end])
    #         else:
    #             c += 1
    #             newPoint.append(point[i])