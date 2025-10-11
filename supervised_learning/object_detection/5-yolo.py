#!/usr/bin/env python3
""" Task 0: 0. Initialize Yolo """
import tensorflow.keras as K  # type: ignore
import numpy as np
import glob
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on the class threshold.

        Args:
            boxes (list of numpy.ndarray): Processed bounding boxes.
            box_confidences (list of numpy.ndarray): Processed box confidences.
            box_class_probs (list of numpy.ndarray):
            Processed box class probabilities.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        # Flatten the lists into single arrays
        boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])
        box_confidences = np.concatenate(
            [confidence.reshape(-1) for confidence in box_confidences])
        box_class_probs = np.concatenate(
            [probs.reshape(-1, probs.shape[-1]) for probs in box_class_probs])

        # Calculate the box scores
        box_scores = box_confidences * np.max(box_class_probs, axis=-1)

        # Filter based on the class threshold
        mask = box_scores >= self.class_t
        filtered_boxes = boxes[mask]
        box_classes = np.argmax(box_class_probs[mask], axis=-1)
        box_scores = box_scores[mask]

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, box2):
        """
        Calculates the Intersection over Union
        (IoU) between two bounding boxes.

        Args:
            box1 (numpy.ndarray): First bounding box [x1, y1, x2, y2].
            box2 (numpy.ndarray): Second bounding box [x1, y1, x2, y2].

        Returns:
            float: IoU between the two boxes.
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        return intersection_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.

        Args:
            filtered_boxes (numpy.ndarray): Filtered bounding boxes.
            box_classes (numpy.ndarray): Class IDs for the filtered boxes.
            box_scores (numpy.ndarray): Scores for the filtered boxes.

        Returns:
            tuple: (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        # Initialize lists to store the final results
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Iterate over each unique class
        for class_id in np.unique(box_classes):
            # Get indices of boxes belonging to the current class
            class_indices = np.where(box_classes == class_id)[0]

            # Extract boxes, scores, and classes for the current class
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            # Sort boxes by score in descending order
            sorted_indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            # Apply NMS
            keep_indices = []
            while len(class_boxes) > 0:
                # Keep the box with the highest score
                keep_indices.append(sorted_indices[0])

                # Compute IoU of the remaining boxes with the highest-scoring
                # box
                ious = np.array([self.iou(class_boxes[0], box)
                                for box in class_boxes[1:]])
                # Remove boxes with IoU > nms_t
                remove_indices = np.where(ious > self.nms_t)[0] + 1
                class_boxes = np.delete(class_boxes, remove_indices, axis=0)
                class_scores = np.delete(class_scores, remove_indices, axis=0)
                sorted_indices = np.delete(
                    sorted_indices, remove_indices, axis=0)

            # Add the kept boxes to the final results
            box_predictions.extend(filtered_boxes[keep_indices])
            predicted_box_classes.extend(box_classes[keep_indices])
            predicted_box_scores.extend(box_scores[keep_indices])

        # Convert lists to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            tuple: (images, image_paths)
        """
        # creating a correct full path argument
        images = []
        image_paths = glob.glob(folder_path + '/*', recursive=False)

        # creating the images list
        for imagepath_i in image_paths:
            images.append(cv2.imread(imagepath_i))

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocesses the images for the Darknet model.

        Args:
            images (list of numpy.ndarray): List of images to preprocess.

        Returns:
            tuple: (pimages, image_shapes)
        """
        dims = []
        res_images = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for image in images:
            dims.append(image.shape[:2])

        dims = np.stack(dims, axis=0)

        newtam = (input_h, input_w)

        interpolation = cv2.INTER_CUBIC
        for image in images:
            resize_img = cv2.resize(image, newtam, interpolation=interpolation)
            resize_img = resize_img / 255
            res_images.append(resize_img)

        res_images = np.stack(res_images, axis=0)

        return (res_images, dims)
