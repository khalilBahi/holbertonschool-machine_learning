#!/usr/bin/env python3
""" Task 0: 0. Initialize Yolo """
import tensorflow.keras as K  # type: ignore
import numpy as np


class Yolo:
    """
    The Yolo class is used for object detection using the YOLOv3 model.
    It initializes with the necessary configurations and loads
    the pre-trained model.

    Attributes:
    model : Keras Model
        The YOLO object detection model loaded from a file.
    class_names : list of str
        A list of the class names used by the model for object detection.
    class_t : float
        The threshold used to filter out objects with a confidence score
        below this value.
    nms_t : float
        The threshold for non-max suppression, used to filter out overlapping
        bounding boxes.
    anchors : numpy.ndarray
        An array of predefined anchor boxes used by YOLO for object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class.

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to the file containing class names.
            class_t (float): Box score threshold
            for the initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes with
            shape (outputs, anchor_boxes, 2).
        """
        # Load the Darknet Keras model
        self.model = K.models.load_model(model_path, compile=False)

        # Load the class names
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]

        # Set the thresholds and anchor boxes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
