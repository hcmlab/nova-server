import numpy as np

from scipy.special import softmax
from imantics import Mask
from shapely.geometry import Point, Polygon


def prediction_to_binary_mask(prediction, label_count):
    binary_masks = []
    for mask in prediction:
        binary_masks_for_class = []

        for label in range(0, label_count):
            binary_masks_for_class.append(np.where(label == mask, True, False))

        binary_masks.append(binary_masks_for_class)

    return binary_masks


def mask_to_polygons(masks):
    polygons_per_mask = []

    for mask in masks:
        polygons = []

        for layer in mask:
            polygons.append(Mask(layer).polygons())
        polygons_per_mask.append(polygons)

    return polygons_per_mask


def get_confidence_from_prediction(predictions, polygons_per_frame):
    # TODO MARCO verbessern
    predictions_in_percentage = softmax(predictions, axis=1)
    frame_polygons_confidences = []
    for frame_id, frame in enumerate(polygons_per_frame):
        label_type_polygons_confidences = []
        for label_type_id, label_type in enumerate(frame):
            polygons_confidences = []
            for polygon in label_type.polygons:
                polygon = Polygon(polygon.reshape(-1, 2))
                width = predictions[0][0].shape[0]
                height = predictions[0][0].shape[1]
                point_counter = 0
                polygon_conf = 0
                for x in range(0, width):
                    for y in range(0, height):
                        if polygon.contains(Point(x, y)) or polygon.intersection(Point(x, y)):
                            point_counter += 1
                            polygon_conf += predictions_in_percentage[frame_id][label_type_id][x, y]
                polygons_confidences.append(polygon_conf/point_counter)
            label_type_polygons_confidences.append(polygons_confidences)
        frame_polygons_confidences.append(label_type_polygons_confidences)

    return frame_polygons_confidences
