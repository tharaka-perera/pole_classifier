import asyncio
import fnmatch
import os

import numpy as np
import pandas as pd
import torch
from geographiclib.geodesic import Geodesic
from torchvision import ops
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from utils.utils import download_images_from_df, get_heading_diff, wrap_around_heading, PanoramaViewer, \
    sort_pano_locations, \
    fetch_metadata_for_pano, process_pano_metadata, select_panos, process_images_with_yolo

api_key = os.getenv("API_KEY")


def find_files_with_partial_name(directory, partial_name):
    matched_files = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, f'*{partial_name}*'):
            matched_files.append(os.path.join(directory, file))
    return matched_files


def load_labels_from_txt(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert from normalized coordinates to absolute coordinates
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            # Add dummy confidence and track_id
            confidence = 1.0
            boxes.append([x_min * 640, y_min * 640, x_max * 640, y_max * 640, confidence, class_id])

    # Convert to numpy array
    boxes_array = torch.as_tensor(boxes)

    # Create Boxes object
    boxes_obj = Boxes(boxes_array, (640, 640))
    return boxes_obj


def is_bbox_within_vertical_center(image_width, bbox):
    x1, y1, x2, y2 = bbox
    vertical_center = image_width / 2

    if x1 <= vertical_center <= x2:
        return True
    return False


def is_heading_within_bbox(bbox, pole_heading, camera_heading, camera_pitch):
    pole_heading = wrap_around_heading(pole_heading)
    x1, y1, x2, y2 = bbox
    viewer = PanoramaViewer(fov=120, width=640, height=640, heading=camera_heading, pitch=camera_pitch)
    start_h = wrap_around_heading(viewer.map(x1, y1)['heading'])
    end_h = wrap_around_heading(viewer.map(x2, y2)['heading'])
    max_heading, min_heading = max(start_h, end_h), min(start_h, end_h)
    if min_heading <= pole_heading <= max_heading:
        if max_heading - min_heading > 180:
            return False
        return True
    return False


def filter_boxes_outside_threshold(boxes, threshold=100):
    left_margin = 320 - threshold
    right_margin = 320 + threshold

    mask = (((left_margin <= boxes.xyxy[:, 0]) & (boxes.xyxy[:, 0] <= right_margin)) |
            ((left_margin <= boxes.xyxy[:, 2]) & (boxes.xyxy[:, 0] <= right_margin)))
    filtered_boxes = boxes[mask]

    return filtered_boxes


def filter_boxes_outside_threshold2(boxes, threshold=50):
    box_center = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2
    left_margin = 320 - threshold
    right_margin = 320 + threshold

    mask = ((box_center <= 320) & (boxes.xyxy[:, 2] > left_margin)) | (
        (box_center > 320) & (boxes.xyxy[:, 0] < right_margin))
    filtered_boxes = boxes[mask]

    return filtered_boxes


def measure_heights_of_boxes(boxes, distance, camera_pitch, camera_heading, fov):
    viewer = PanoramaViewer(fov=fov, width=640, height=640, heading=camera_heading, pitch=camera_pitch)

    # Extract coordinates
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Map coordinates to pitches
    pitch_1 = np.array([viewer.map(x, y)['pitch'] for x, y in zip(x1, y1)]) * np.pi / 180
    pitch_2 = np.array([viewer.map(x, y)['pitch'] for x, y in zip(x2, y2)]) * np.pi / 180

    # Calculate top and bottom pitches
    pitch_top = np.maximum(pitch_1, pitch_2)
    pitch_bot = np.minimum(pitch_1, pitch_2)

    # Calculate heights
    heights = distance * (np.tan(pitch_top) - np.tan(pitch_bot))
    return heights


def remove_outliers_by_height(boxes, box_heights, min_value, max_value):
    mask = (min_value <= box_heights) & (box_heights <= max_value)
    return box_heights[mask], boxes[mask]


def height_based_nms(boxes, iou_threshold):
    if len(boxes) == 0:
        return boxes

    heights = boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
    normalized_heights = heights / heights.max()
    return boxes[ops.nms(boxes.xyxy, normalized_heights, iou_threshold)]


def find_poles_within_radius(dataset, pano_location, radius):
    # Extract the latitude and longitude of the panorama location
    pano_lat, pano_lon, pano_heading, index = pano_location

    def calculate_distance_and_heading(row):
        pole_lat, pole_lon = (row['pole_lat'], row['pole_lon'])
        geodesic_result = Geodesic.WGS84.Inverse(pano_lat, pano_lon, pole_lat, pole_lon)
        heading_diff = get_heading_diff(pano_heading, geodesic_result['azi1'])
        return geodesic_result['s12'], wrap_around_heading(geodesic_result['azi1']), heading_diff

    # Apply the distance calculation to each row
    dataset[['distance', 'heading', 'heading_diff']] = dataset.apply(
        lambda row: pd.Series(calculate_distance_and_heading(row)), axis=1)

    # Filter the DataFrame to include only rows within the specified radius
    poles_within_radius = dataset[
        (dataset['distance'] <= radius) & (dataset['heading_diff'] <= 70) & (dataset['index'] != index)]

    return poles_within_radius


if __name__ == "__main__":
    # fetch meta-data from google using the location data and save to a csv file
    fetch_metadata_for_pano('../data/resource_features_2.csv', '../data/cronulla_source_panoramas.csv')

    # filter too close or distance panos, too old panos and remove unnecessary columns from csv file
    df = pd.read_csv('../data/cronulla_source_panoramas.csv')
    df = process_pano_metadata(df)
    df.to_csv('../data/filtered_cronulla_panoramas.csv', index=False)

    # sort panoramas according to year and heading differences
    df = pd.read_csv('../data/filtered_cronulla_panoramas.csv')
    df = sort_pano_locations(df)
    df.to_csv('../data/sorted_cronulla_panoramas.csv', index=False)

    # select only 3 panoramas from each pole to download from the sorted list
    select_panos(3, '../data/sorted_cronulla_panoramas.csv', '../data/downloaded_cronulla_panoramas.csv')

    # download initial images asynchronously - make sure that the output directory is created
    asyncio.run(download_images_from_df('../data/downloaded_cronulla_panoramas.csv', '../data/source_cronulla_images'))

    # download zoomed images and save calculated meta-data for images in a csv
    model = YOLO('../models/bestv10.pt')
    processed_df = asyncio.run(
        process_images_with_yolo(model, '../data/downloaded_cronulla_panoramas.csv', '../data/source_cronulla_images',
                                 '../data/zoomed_cronulla_images', 1000))
    processed_df.to_csv('../data/zoomed_panoramas_cronulla.csv', index=False)

    # optional step to download images for pole material identification
    # asyncio.run(download_material_images(model, '../data/zoomed_panoramas_brighten.csv', '../data/steel-poles/source'))
