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

    def sigmoid(self, x):
        """Applies the sigmoid function to an input."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the Darknet model.

        Args:
            outputs (list of numpy.ndarray):
            Predictions from the Darknet model.
            image_size (numpy.ndarray):
            Original image size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box coordinates (t_x, t_y, t_w, t_h)
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Apply sigmoid to t_x and t_y to get box center relative to grid
            # cell
            b_x = self.sigmoid(t_x)
            b_y = self.sigmoid(t_y)

            # Apply exponential to t_w and t_h and scale by anchor box
            # dimensions
            b_w = np.exp(t_w) * self.anchors[i, :, 0]
            b_h = np.exp(t_h) * self.anchors[i, :, 1]

            # Create grid indices
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Calculate absolute box coordinates
            abs_x = (b_x + grid_x) / grid_width
            abs_y = (b_y + grid_y) / grid_height
            abs_w = b_w / self.model.input.shape[1]  # Normalize by input width
            # Normalize by input height
            abs_h = b_h / self.model.input.shape[2]

            # Convert to (x1, y1, x2, y2) format
            x1 = (abs_x - abs_w / 2) * image_size[1]
            y1 = (abs_y - abs_h / 2) * image_size[0]
            x2 = (abs_x + abs_w / 2) * image_size[1]
            y2 = (abs_y + abs_h / 2) * image_size[0]

            # Stack coordinates into a single array
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Extract box confidence and apply sigmoid
            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Extract class probabilities and apply sigmoid
            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
