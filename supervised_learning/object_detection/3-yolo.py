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
        # Handle empty input gracefully
        if filtered_boxes.size == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        box_predictions = []  # collected boxes after NMS
        predicted_box_classes = []  # corresponding class ids
        predicted_box_scores = []  # corresponding scores

        # Perform NMS per class
        for c in np.unique(box_classes):
            # indices for this class
            idxs = np.where(box_classes == c)[0]
            boxes_c = filtered_boxes[idxs]
            scores_c = box_scores[idxs]

            # sort descending by score
            order = scores_c.argsort()[::-1]
            boxes_c = boxes_c[order]
            scores_c = scores_c[order]

            while boxes_c.shape[0] > 0:
                # select top-scoring box
                box_predictions.append(boxes_c[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(scores_c[0])

                if boxes_c.shape[0] == 1:
                    break  # nothing else to compare

                # Compute IoU of the top box with the rest (vectorized)
                xx1 = np.maximum(boxes_c[0, 0], boxes_c[1:, 0])
                yy1 = np.maximum(boxes_c[0, 1], boxes_c[1:, 1])
                xx2 = np.minimum(boxes_c[0, 2], boxes_c[1:, 2])
                yy2 = np.minimum(boxes_c[0, 3], boxes_c[1:, 3])

                inter_w = np.maximum(0, xx2 - xx1)
                inter_h = np.maximum(0, yy2 - yy1)
                inter_area = inter_w * inter_h

                area0 = ((boxes_c[0, 2] - boxes_c[0, 0]) *
                         (boxes_c[0, 3] - boxes_c[0, 1]))
                areas = ((boxes_c[1:, 2] - boxes_c[1:, 0]) *
                         (boxes_c[1:, 3] - boxes_c[1:, 1]))
                union = area0 + areas - inter_area
                iou = np.zeros_like(inter_area)
                valid = union > 0
                iou[valid] = inter_area[valid] / union[valid]

                # keep boxes with IoU < threshold (suppress if >= threshold)
                keep = np.where(iou < self.nms_t)[0]
                boxes_c = boxes_c[keep + 1]
                scores_c = scores_c[keep + 1]

        # Convert to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        # Order by class then by score (descending) inside each class
        # Use lexsort with negative scores for descending
        order = np.lexsort((-predicted_box_scores, predicted_box_classes))
        box_predictions = box_predictions[order]
        predicted_box_classes = predicted_box_classes[order]
        predicted_box_scores = predicted_box_scores[order]

        return (box_predictions, predicted_box_classes, predicted_box_scores)
