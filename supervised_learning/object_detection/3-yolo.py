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
        with open(classes_path, "r") as file:
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

        for i in range(len(outputs)):
            grid_h = outputs[i].shape[0]
            grid_w = outputs[i].shape[1]
            anchor_boxes = outputs[i].shape[2]

            boxes_i = outputs[i][..., 0:4]

            for anchor_n in range(anchor_boxes):
                for cy in range(grid_h):
                    for cx in range(grid_w):
                        tx = outputs[i][cy, cx, anchor_n, 0]
                        ty = outputs[i][cy, cx, anchor_n, 1]
                        tw = outputs[i][cy, cx, anchor_n, 2]
                        th = outputs[i][cy, cx, anchor_n, 3]

                        pw = self.anchors[i][anchor_n][0]
                        ph = self.anchors[i][anchor_n][1]

                        bx = self.sigmoid(tx) + cx
                        by = self.sigmoid(ty) + cy

                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)

                        bx /= grid_w
                        by /= grid_h
                        bw /= int(self.model.input.shape[1])
                        bh /= int(self.model.input.shape[2])

                        x1 = (bx - bw / 2) * image_size[1]
                        y1 = (by - bh / 2) * image_size[0]
                        x2 = (bx + bw / 2) * image_size[1]
                        y2 = (by + bh / 2) * image_size[0]

                        boxes_i[cy, cx, anchor_n, 0] = x1
                        boxes_i[cy, cx, anchor_n, 1] = y1
                        boxes_i[cy, cx, anchor_n, 2] = x2
                        boxes_i[cy, cx, anchor_n, 3] = y2

            boxes.append(boxes_i)

            confidence_i = self.sigmoid(outputs[i][..., 4:5])
            box_confidences.append(confidence_i)

            probs_i = self.sigmoid(outputs[i][..., 5:])
            box_class_probs.append(probs_i)

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
        scores = []

        for bc, probs in zip(box_confidences, box_class_probs):
            scores.append(bc * probs)

        box_classes_list = []
        box_scores_list = []

        for score in scores:
            box_classes_list.append(np.argmax(score, axis=-1).flatten())
            box_scores_list.append(np.max(score, axis=-1).flatten())

        boxes = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes, axis=0)
        box_classes = np.concatenate(box_classes_list, axis=0)
        box_scores = np.concatenate(box_scores_list, axis=0)

        mask = np.where(box_scores >= self.class_t)

        return boxes[mask], box_classes[mask], box_scores[mask]

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
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in set(box_classes):
            idx = np.where(box_classes == c)

            boxes_c = filtered_boxes[idx]
            scores_c = box_scores[idx]
            classes_c = box_classes[idx]

            x1 = boxes_c[:, 0]
            y1 = boxes_c[:, 1]
            x2 = boxes_c[:, 2]
            y2 = boxes_c[:, 3]

            area = (x2 - x1) * (y2 - y1)
            sorted_idx = np.flip(scores_c.argsort(), axis=0)

            keep = []
            while len(sorted_idx) > 0:
                i = sorted_idx[0]
                keep.append(i)

                if len(sorted_idx) == 1:
                    break

                remaining = sorted_idx[1:]

                xx1 = np.maximum(x1[i], x1[remaining])
                yy1 = np.maximum(y1[i], y1[remaining])
                xx2 = np.minimum(x2[i], x2[remaining])
                yy2 = np.minimum(y2[i], y2[remaining])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)

                intersection = w * h
                union = area[i] + area[remaining] - intersection
                iou = intersection / union

                below_threshold = np.where(iou <= self.nms_t)[0]
                sorted_idx = sorted_idx[below_threshold + 1]

            keep = np.array(keep)
            box_predictions.append(boxes_c[keep])
            predicted_box_classes.append(classes_c[keep])
            predicted_box_scores.append(scores_c[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
