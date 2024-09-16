import asyncio
import datetime as dt
import math
import os

import pandas as pd
from geographiclib.geodesic import Geodesic
from utils.streetview import SVClient
from utils.streetview_private_api import SVPrivateClient
from utils.utils import read_images_from_folder, get_center_hp, download_images_from_df, calculate_heading, \
    get_heading_diff, wrap_around_heading, PanoramaViewer, get_fov_based_on_distance, sort_pano_locations, fetch_metadata_for_pano, process_pano_metadata
from ultralytics import YOLOv10
from ultralytics.engine.results import Boxes
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import ops
import numpy as np

api_key = os.getenv("API_KEY")


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


async def process_images_with_yolo(model, csv_path, image_folder):
    svc = SVClient(api_key=api_key, max_concurrent_requests=50)
    df = pd.read_csv(csv_path).head(1000)
    idx = 0
    prev_index = 0
    for _row_id, row in df.iterrows():
        index = row['index']
        pano_id = row['pano_id']
        year = row['year']
        distance = row['distance']
        heading = row['heading']
        pitch = row['pitch']
        fov = row['fov']
        filename = f"{index}_id_{idx}_d_{round(distance)}_{pano_id}_y_{year}.jpg"
        if index == prev_index:
            idx += 1
        else:
            idx = 0
        prev_index = index
        full_path = os.path.join(image_folder, filename)

        if os.path.exists(full_path):
            """
            when providing to for inference YOLO, it should be in BGR format;
            see https://github.com/ultralytics/ultralytics/issues/9912
            """
            image_bgr = cv2.imread(full_path)
            result = model.predict(source=image_bgr, conf=0.2, iou=0.15, verbose=False, device='cpu')
            boxes = filter_boxes_outside_threshold2(result[0].boxes)
            boxes = height_based_nms(boxes, 0.2)
            # result[0].update(boxes=boxes.data)
            box_heights = measure_heights_of_boxes(boxes.xyxy, distance, pitch, heading, fov)
            box_heights, boxes = remove_outliers_by_height(boxes, box_heights, 4.5, 27)
            height_errors = (box_heights - 9.47) ** 2
            # result[0].save(filename=os.path.join('../data/cronulla_pole_detector_out', filename))

            pole_coordinates = None

            if len(boxes) > 0:
                min_error_index = height_errors.argmin()
                # for box in result[0].boxes:
                #     x1, y1, x2, y2 = box.xyxy[0].tolist()
                #     color = (255, 0, 0)  # blue in bgr
                #     cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # for i, box in enumerate(boxes):
                #     x1, y1, x2, y2 = box.xyxy[0].tolist()
                #     if i == min_error_index:
                #         color = (0, 0, 255)  # red in bgr
                #         cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # cv2.imwrite(os.path.join('../data/bounding_box_2', filename), image_bgr)
                pole_coordinates = boxes[min_error_index].xyxy[0].tolist()
            if len(boxes) == 0:  # there are no poles detected
                print(f'No poles detected in image {_row_id}_{idx}')
                # result[0].save(filename=os.path.join('../data/cronulla_no_poles_detected', filename))
            # elif len(boxes) > 1:  # multiple poles detected
            #     centered_boxes = 0
            #     for box in boxes:  # check how many
            #         if is_bbox_within_vertical_center(640, box.xyxy[0].tolist()):
            #             centered_boxes += 1
            #     if centered_boxes > 1:
            #         print(f'{centered_boxes} poles detected in image {idx}')  # just take the box with the maximum prob
            #         pole_coordinates = boxes[boxes.conf.argmax()].xyxy[0].tolist()
            #     elif centered_boxes == 0:
            #         # print(f'{len(boxes)} non-centered poles detected in image {idx}')
            #         nearby_poles = find_poles_within_radius(df, row[['lat', 'lon', 'heading', 'index']], 40)
            #         # nearby_poles = nearby_poles.drop(_row_id)
            #         print(f'Found {len(nearby_poles)} poles within 40 meters')
            #         if not nearby_poles.empty:
            #             boxes_to_keep = []
            #             for box in boxes:
            #                 keep_box = True
            #                 for _, pole in nearby_poles.iterrows():
            #                     if is_heading_within_bbox(box.xyxy[0].tolist(), pole['heading'], heading, pitch):
            #                         keep_box = False
            #                         break
            #                 if keep_box:
            #                     boxes_to_keep.append(box)
            #             if len(boxes_to_keep) > 1:
            #                 print(f'{len(boxes_to_keep)} off-centered poles detected in image {idx}')
            #             elif len(boxes_to_keep) == 1:
            #                 pole_coordinates = boxes_to_keep[0].xyxy[0].tolist()
            #                 print(f'One off-centered pole detected in image {index}_{idx}')
            #             else:
            #                 print('No off-centered poles detected')
            #             # pole_coordinates = nearby_poles.iloc[0][['x1', 'y1', 'x2', 'y2']].values
            #         else:
            #             print('No poles found within 40 meters')
            # else:  # only one pole detected
            #     pole_coordinates = boxes.xyxy[0].tolist()
                # print(f'One pole detected in image {idx}')

            if pole_coordinates is not None:
                r_head, r_pitch, r_fov = get_center_hp(*pole_coordinates, heading, pitch, fov=fov)
                await asyncio.create_task(svc.download_and_save_image(pano_id, r_head, r_pitch, r_fov, os.path.join('../data/zoomed_brighten_images', filename)))
            else:
                pass
        else:
            print(f"Image {filename} not found in {image_folder}")


if __name__ == "__main__":
    # fetch_metadata_for_pano('../data/resource_features_2.csv', '../data/cronulla_source_panoramas.csv')
    # df = pd.read_csv('../data/cronulla_source_panoramas.csv')
    # df = process_pano_metadata(df)
    # df.to_csv('../data/filtered_cronulla_panoramas.csv', index=False)
    # df = pd.read_csv('../data/filtered_cronulla_panoramas.csv')
    # df = sort_pano_locations(df)
    # df.to_csv('../data/sorted_cronulla_panoramas.csv', index=False)
    # df = pd.read_csv('../data/sorted_cronulla_panoramas.csv')
    # unique_indices = df['index'].unique()
    # selected_panos = []
    #
    # for idx in unique_indices:
    #     panos = df[df['index'] == idx]
    #     if len(panos) < 3:
    #         panos = df[df['index'] == idx]
    #     else:
    #         panos = panos.head(3)
    #     selected_panos.extend(panos.to_dict('records'))
    # df = pd.DataFrame(selected_panos)
    # df['fov'] = df['distance'].apply(get_fov_based_on_distance)
    # df.to_csv('../data/downloaded_cronulla_panoramas.csv', index=False)
    model = YOLOv10('../models/bestv10.pt')
    # images = read_images_from_folder('../data/images')
    asyncio.run(process_images_with_yolo(model, '../data/downloaded_panoramas.csv', '../data/source_brighten_images'))

    # asyncio.run(download_images_from_df('../data/downloaded_panoramas.csv', '../data/source_brighten_images'))
