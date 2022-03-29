### all functions
import os
import logging
import numpy as np
import pandas as pd
import cv2
import dlib

import face_recognition_models
from collections import OrderedDict
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, path=None, clevel = logging.DEBUG, Flevel = logging.INFO):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set cmd logging
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        self.logger.addHandler(sh)
        # set file logging
        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(Flevel)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

# face detector: hog
face_detector = dlib.get_frontal_face_detector()
# face detector: cnn
cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)
# landmark predictor
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
# face encoder model
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
# cnn face encoder model
cnn_face_recognition_model = face_recognition_models.face_recognition_model_location()
cnn_face_encoder = dlib.face_recognition_model_v1(cnn_face_recognition_model)

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose_vertical", (27, 31)),
    ("nose_horizontal", (31, 36)),
	("jaw", (0, 17))
])

## convert a dlib rectangle object to (top, right, bottom, left)
def rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

## convert a dlib rectangle object to (x, y, w, h) = (left, top, right-left, bottom-top)
def rect_to_bb(rect):
    return rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()

## convert (top, right, bottom, left) to a dlib rectangle object
def css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

## trim operation: make sure face rectangle is within the bounds of image
def trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

## get face locations: return dlib rectangle object
def get_face_locations(image, model='hog'):
    if model == 'cnn':
        return cnn_face_detector(image)
    else:
        return face_detector(image)

## get face landmarks: return a list of landmark sets in the image
def get_face_landmarks(image, face_locations):
    return [pose_predictor_68_point(image, face_location) for face_location in face_locations]

## get face encoding: return a list of face encodings in the image
def get_face_encodings(image, face_locations):
    # need face landmarks
    landmarks = get_face_landmarks(image, face_locations)
    return [np.array(face_encoder.compute_face_descriptor(image, landmark)) for landmark in landmarks]

## calculate distance of an input face to a list of faces
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

## compare faces by distance between face encodings
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    # return a list of bool
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


## convert dlib shape object to a (68, 2) matrix
def landmarks_to_np(landmarks, dtype="int"):
	coords = np.zeros((landmarks.num_parts, 2), dtype=dtype)
	for i in range(0, landmarks.num_parts):
		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

## align face: rotate by eyes connection line and translate by eyes center
def face_align(image, landmarks, desiredLeftEye = (0.35, 0.35), desiredFaceWidth = 256):
    if not isinstance(landmarks, np.ndarray):
        landmarks = landmarks_to_np(landmarks)
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    leftEyePts = landmarks[lStart:lEnd]
    rightEyePts = landmarks[rStart:rEnd]
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    desiredFaceHeight = desiredFaceWidth
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    eyesCenter = int(eyesCenter[0]), int(eyesCenter[1])
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # output: aligned face image (x center is eyesCenter[0])
    return output

## composite left-left and right-right composite faces by cv2.flip
def face_composite(face):
    xcenter = int(face.shape[0] // 2)
    # face: image containing an aligned face (eyesCenter[0] in the center of x axis)
    flipped = cv2.flip(face, 1)  # 1: horizontal, 0: vertical, -1: horizontal + vertical
    left2 = np.hstack((face[:, :xcenter, :], flipped[:, xcenter:, :]))
    right2 = np.hstack((flipped[:, :xcenter, :], face[:, xcenter:, :]))
    return left2, right2

## draw landmarks on faces
def draw_landmarks(image_file, draw_benchmarks=True, save=False):

    image = cv2.imread(image_file)

    face_locations = get_face_locations(image)
    landmarks = get_face_landmarks(image, face_locations)  # list of dlib landmark sets
    landmarks = landmarks_to_np(landmarks[0])  # (68, 2)

    point_size = 1
    point_color = (0, 0, 255)
    thickness = 2
    for point in landmarks:
        cv2.circle(image, point, point_size, point_color, thickness)

    if draw_benchmarks:
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = landmarks[lStart:lEnd]
        rightEyePts = landmarks[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        cv2.circle(image, leftEyeCenter, 1, (0, 255, 0), 3)
        cv2.circle(image, rightEyeCenter, 2, (0, 255, 0), 3)
        cv2.line(image, leftEyeCenter, rightEyeCenter, (255, 0, 0), 2)

    if save:
        cv2.imwrite(save, image)
    else:
        cv2.imshow('Original image with landmarks', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)






