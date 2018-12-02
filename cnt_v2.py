import numpy as np
import cv2
import math
import pickle


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


def remove_bg(frame, hist):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([frameHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cv2.filter2D(dst, -1, disc, dst)

    # smooth
    blur = cv2.GaussianBlur(dst, (3, 3), 0)
    blur = cv2.medianBlur(blur, 25)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)

    # threshold and binary AND
    thresh = cv2.threshold(blur, 100, 255, 0)[1]
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #                               11, 2)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(frame, thresh)
    res = np.hstack((thresh, res))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)  # always change thresh to grayscale for contour
    return res, thresh


def segment_hand(thresh):
    # get contours
    _, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 1)

    # check if contours were found
    if len(cnt) == 0:
        return
    else:
        segment = max(cnt, key=cv2.contourArea)
        return segment


def find_centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        return cx, cy
    else:
        return None


def find_tips(defect, contour, frame, centroid):
    if defect is not None:

        # counter for amount of fingers
        x, y, w, h = cv2.boundingRect(contour)
        count = 0
        possibly_zero = True
        for i in range(defect.shape[0]):
            s, e, f, d = defect[i][0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            distance = math.sqrt((centroid[0] - far[0]) ** 2 + (centroid[1] - far[1]) ** 2)

            if far[1] < centroid[1] and distance > 0.4 * h:
                possibly_zero = False

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57  # cosine theorem
            if angle <= 100:  # angle less than 100 degree, treat as fingers
                count += 1
                cv2.circle(frame, end, 8, [211, 84, 0], -1)
                cv2.line(frame, centroid, end, [0, 255, 0], 2)

                # in order to not draw lines right next to each other, set a threshold distance
                distance = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
                if distance > 100:
                    cv2.circle(frame, start, 8, [211, 84, 0], -1)
                    cv2.line(frame, centroid, start, [0, 255, 0], 2)
        gesture = min([5, count])
        if gesture <= 1 and possibly_zero:
            gesture = 0
        return gesture


def main():
    hist = get_hand_hist()  # get skin color

    # camera
    cam = cv2.VideoCapture(0)
    count = ' '
    while True:

        frame = cam.read()[1]
        frame = cv2.flip(frame, 1)  # horizontal flip
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing
        res, thresh = remove_bg(frame, hist)  # remove everything that isn't skin color

        hand = segment_hand(thresh)  # get hand segment

        # get centroid
        centroid = find_centroid(hand)
        cv2.circle(frame, centroid, 5, [255, 0, 255], -1)

        # check for hand segment
        if hand is not None:
            seg = hand
            cv2.drawContours(frame, seg, -1, (0, 0, 255), 2)

            hull = cv2.convexHull(hand, returnPoints=False)
            defects = cv2.convexityDefects(hand, hull)
            # far = f_tips(defects, hand, centroid)
            # cv2.circle(frame, far, 5, [255, 0, 0], -1)
            print(find_tips(defects, hand, frame, centroid))
            count = find_tips(defects, hand, frame, centroid)

        res3 = np.hstack((frame, res))
        cv2.putText(res3, str(count), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0))
        cv2.imshow('frame and thresh', res3)


        k = cv2.waitKey(10)
        if k == 27:
            break

    cv2.destroyAllWindows()


main()
