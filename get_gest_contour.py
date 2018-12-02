import numpy as np
import cv2
import pickle
import os, os.path
from os import path


def make_folder():
    if not os.path.exists("gesture"):
        os.mkdir("gesture")


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


def store_image(filename, filepath):
    if path.exists(filepath + filename + '.jpg'):
        choice = raw_input("Name already exists. Would you like to change the name? (y/n):\t")
        if choice.lower() == 'y':
            filename = raw_input("Name:\t")

        else:
            print("\nOVERWRITING\n")

    return filename


def get_image():
    filepath = 'C:/Users/avald/PycharmProjects/final/gesture/'
    crop = []
    x, y, w, h = 100, 100, 250, 250
    cap = cv2.VideoCapture(0)
    hist = get_hand_hist()

    # filter things

    while True:
        frame = cap.read()[1]
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (x-3, y-3), (x + w + 3, y + h + 3), (255, 0, 0), 2)
        # cv2.imshow('mask', )
        cv2.imshow('frame', frame)

        keypress = cv2.waitKey(1)
        # pressing 'c' generates a crop from the rectangle
        if keypress == ord('c'):
            crop = frame[y:y + h, x:x + w]
            cv2.imshow("cropped", crop)
        # pressing 's' saves the cropped image if there is one
        # otherwise it alerts you to say that nothing has been cropped
        elif keypress == ord('s'):
            array = np.array(crop)
            if array.size == 0:
                print("Nothing has been cropped")
            if array.size > 0:
                filename = raw_input('Name: ')
                filename = store_image(filename, filepath)
                status = cv2.imwrite(filepath
                                     + filename + '.jpg', crop)

                cv2.destroyWindow("cropped")
                print(status)
        elif keypress == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


make_folder()
get_image()
