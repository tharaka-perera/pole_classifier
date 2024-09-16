import asyncio
import datetime as dt
import glob
import os
from io import BytesIO
from typing import List, Tuple, Optional
import math
import numpy as np

import aiofiles
import cv2
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from geographiclib.geodesic import Geodesic
from pydantic import BaseModel
from streetview import search_panoramas

load_dotenv()

api_key = os.getenv("API_KEY")


def request_concurrency_limit(max_concurrent_requests: int):
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    def decorator(func):
        def wrapper(*args, **kwargs):
            async def sem_task():
                async with semaphore:
                    return func(*args, **kwargs)

            return asyncio.run(sem_task())

        return wrapper

    return decorator


def save_dataframe_to_file(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)


async def save_image_async(image: Image.Image, filename: str):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    async with aiofiles.open(filename, 'wb') as out_file:
        await out_file.write(img_byte_arr.read())


def fetch_metadata_for_pano(source, destination):
    # coordinates = [('1', -37.917951435480600, 144.98947978019700), ('2', -37.919236, 144.990561)]
    coordinates = filter_poles_by_confidence(pd.read_csv(source), 0.5)
    coordinates = read_nearmaps_data_to_coords(coordinates)
    panoramas_df = asyncio.run(get_panoramas_for_coordinates(coordinates))
    panoramas_df.to_csv(destination, index=False)


class Panorama(BaseModel):
    index: int
    pole_id: str
    pole_lat: float
    pole_lon: float
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float]
    roll: Optional[float]
    date: Optional[str]
    elevation: Optional[float]


def panoramas_to_dataframe(panoramas: List[Panorama]) -> pd.DataFrame:
    data = [pano.model_dump() for pano in panoramas]
    df = pd.DataFrame(data)
    return df


async def get_panoramas_for_coordinates(coords: List[Tuple[str, float, float]]) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, search_panoramas, lat, lon) for _, lat, lon in coords]
    all_panoramas = await asyncio.gather(*tasks)

    panoramas_with_pole_id = []
    for idx, ((pole_id, lat, lon), panos) in enumerate(zip(coords, all_panoramas)):
        for pano in panos:
            pano = Panorama(index=idx, pole_id=pole_id, pole_lat=lat, pole_lon=lon, **pano.dict())
            panoramas_with_pole_id.append(pano)

    return panoramas_to_dataframe(panoramas_with_pole_id)


def read_nearmaps_data_to_coords(df):
    # df = pd.read_csv(file_path)
    coords = [(row['id'], row['lat'], row['lon']) for _, row in df.iterrows()]
    return coords


def filter_poles_by_confidence(df, confidence=0.5):
    return df[df['confidence'] >= confidence]


#############


def filter_old_panos(end_year: dt.datetime.year, df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[df['date'].isna() | (df['date'].dt.year >= end_year)]


def wrap_around_heading(heading):
    if heading < 0:
        heading = heading + 360
    elif heading > 360:
        heading = heading - 360
    return heading


def calculate_heading(coord1, coord2):
    heading = Geodesic.WGS84.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])['azi1']
    if heading < 0:
        heading = heading + 360
    elif heading > 360:
        heading = heading - 360
    return heading


def calculate_pitch(distance):
    # gradient from linear interpolation 3 / (4.1 - 5.9)
    pitch = -1.6666 * distance + 24.833
    return max(pitch, 0)


def calculate_distances_and_heading(df):
    distances = []
    unique_indices = df['index'].unique()

    for idx in unique_indices:
        pole_coords = df[df['index'] == idx][['pole_lat', 'pole_lon']].iloc[0]
        pole_coords = (pole_coords['pole_lat'], pole_coords['pole_lon'])

        for _, row in df[df['index'] == idx].iterrows():
            pano_coords = (row['lat'], row['lon'])
            distance = Geodesic.WGS84.Inverse(pano_coords[0], pano_coords[1], pole_coords[0], pole_coords[1])['s12']
            heading = calculate_heading(pano_coords, pole_coords)
            pitch = calculate_pitch(distance)
            distance_entry = row.to_dict()
            distance_entry['distance'] = distance
            distance_entry['heading'] = heading
            distance_entry['pitch'] = pitch
            distances.append(distance_entry)

    return pd.DataFrame(distances)


def filter_panos_by_distance(df, min_distance, max_distance):
    return df[df['distance'].between(min_distance, max_distance)]


def process_pano_metadata(df):
    df = df.copy()
    df = filter_old_panos(2017, df)
    df.loc[:, 'date'] = df['date'].fillna(pd.Timestamp('2024-01-01'))
    df.loc[:, 'year'] = df['date'].dt.year
    df = calculate_distances_and_heading(df)
    df = filter_panos_by_distance(df, 3, 15)
    df = df.sort_values(by=['index', 'distance'])
    df = df[
        ['index', 'pole_id', 'pole_lat', 'pole_lon', 'pano_id', 'lat', 'lon', 'heading', 'pitch', 'year', 'distance']]
    return df


def get_heading_diff(heading1, heading2):
    diff = abs(heading1 - heading2)
    if diff > 180:
        diff = 360 - diff
    return diff


def sort_pano_locations(df, heading_threshold=15):
    unique_indices = df['index'].unique()
    selected_panos = []

    for idx in unique_indices:
        panos = df[df['index'] == idx]
        keep_panos = []
        backup_panos = []

        for i in range(len(panos)):
            current_pano = panos.iloc[i]
            add_to_backup = False

            for s_id, selected_pano in enumerate(keep_panos):
                heading_diff = get_heading_diff(selected_pano['heading'], current_pano['heading'])
                if heading_diff < heading_threshold:
                    if selected_pano['year'] < current_pano['year']:
                        current_pano, keep_panos[s_id] = keep_panos[s_id], current_pano
                    add_to_backup = True
                    break

            if add_to_backup:
                backup_panos.append(current_pano)
            else:
                keep_panos.append(current_pano)

        selected_panos.extend(keep_panos)
        selected_panos.extend(backup_panos)

    return pd.DataFrame(selected_panos)


def get_fov_based_on_distance(distance):
    if distance < 7:
        return 120
    elif distance < 8:
        return 115
    elif distance < 9:
        return 110
    elif distance < 10:
        return 100
    elif distance < 15:
        return 90
    return 80


async def download_images_from_df(filepath, output_dir):
    df = pd.read_csv(filepath)
    df['fov'] = df['distance'].apply(get_fov_based_on_distance)
    from .streetview import SVClient
    svc = SVClient(api_key, 50)
    await svc.download_and_save_images(df.head(1000), output_dir)


# fetch_metadata_for_pano('../data/resource_features_202311171040.csv', '../data/panoramas.csv')
# df = pd.read_csv('../data/panoramas.csv')
# df = process_pano_metadata(df)
# df.to_csv('../data/filtered_panoramas.csv', index=False)
# df = pd.read_csv('../data/filtered_panoramas.csv')
# df = sort_pano_locations(df)
# df = pd.read_csv('../data/sorted_panoramas.csv')
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
# df.to_csv('../data/downloaded_panoramas.csv', index=False)


# Download zoomed images

def read_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    images = [cv2.imread(file)[..., ::-1] for file in image_files]  # OpenCV image (BGR to RGB)
    return images


def heading_pitch_to_vector(heading, pitch):
    heading_rad = np.radians(heading)
    pitch_rad = np.radians(pitch)
    x = np.cos(pitch_rad) * np.sin(heading_rad)
    y = np.cos(pitch_rad) * np.cos(heading_rad)
    z = np.sin(pitch_rad)
    return np.array([x, y, z])


def vector_to_heading_pitch(vector):
    x, y, z = vector
    heading = np.degrees(np.arctan2(x, y))
    pitch = np.degrees(np.arcsin(z))
    return heading, pitch


def average_heading_pitch(heading_min, pitch_min, heading_max, pitch_max):
    vec_min = heading_pitch_to_vector(heading_min, pitch_min)
    vec_max = heading_pitch_to_vector(heading_max, pitch_max)

    # Average the vectors
    avg_vec = (vec_min + vec_max) / 2

    # Normalize the averaged vector
    avg_vec /= np.linalg.norm(avg_vec)

    # Convert the vector back to heading and pitch
    avg_heading, avg_pitch = vector_to_heading_pitch(avg_vec)
    return avg_heading, avg_pitch


def vector_angle(v1, v2):
    """Calculate the angular distance between two 3D vectors in radians."""
    dot_product = np.dot(v1, v2)
    # Ensure the value is within the valid range for acos due to numerical precision
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return angle


def calculate_fov_for_bounding_box(heading_min, pitch_min, heading_max, pitch_max):
    """Calculate the FOV needed to zoom into the bounding box defined by the given headings and pitches."""
    # Convert bounding box corners to 3D vectors
    center_heading = (heading_min + heading_max) / 2
    vec_min = heading_pitch_to_vector(center_heading, pitch_min)
    vec_max = heading_pitch_to_vector(center_heading, pitch_max)
    required_fov = np.degrees(vector_angle(vec_min, vec_max)) * 1.08

    # Calculate the angular distance (in radians) between the vectors for width and height
    # angular_width = vector_angle(
    #     heading_pitch_to_vector(heading_min, (pitch_min + pitch_max) / 2),
    #     heading_pitch_to_vector(heading_max, (pitch_min + pitch_max) / 2)
    # )
    # angular_height = vector_angle(
    #     heading_pitch_to_vector((heading_min + heading_max) / 2, pitch_min),
    #     heading_pitch_to_vector((heading_min + heading_max) / 2, pitch_max)
    # )
    #
    # # Convert the angular width and height from radians to degrees
    # angular_width_deg = np.degrees(angular_width)
    # angular_height_deg = np.degrees(angular_height)
    # required_fov = max(angular_width_deg, angular_height_deg) * 1.05

    return required_fov


def sgn(x):
    return (x > 0) - (x < 0)


class PanoramaViewer:
    def __init__(self, fov, width, height, heading, pitch):
        self.fov = fov
        self.width = width
        self.height = height
        self.heading = heading
        self.pitch = pitch

    def unmap(self, heading, pitch):
        fov = self.fov * np.pi / 180.0
        width = self.width
        height = self.height

        f = 0.5 * width / np.tan(0.5 * fov)

        h = heading * np.pi / 180.0
        p = pitch * np.pi / 180.0

        x = f * np.cos(p) * np.sin(h)
        y = f * np.cos(p) * np.cos(h)
        z = f * np.sin(p)

        h0 = self.heading * np.pi / 180.0
        p0 = self.pitch * np.pi / 180.0

        x0 = f * np.cos(p0) * np.sin(h0)
        y0 = f * np.cos(p0) * np.cos(h0)
        z0 = f * np.sin(p0)

        t = f * f / (x0 * x + y0 * y + z0 * z)

        ux = sgn(np.cos(p0)) * np.cos(h0)
        uy = - sgn(np.cos(p0)) * np.sin(h0)
        uz = 0

        vx = -np.sin(p0) * np.sin(h0)
        vy = -np.sin(p0) * np.cos(h0)
        vz = np.cos(p0)

        x1 = t * x
        y1 = t * y
        z1 = t * z

        dx10 = x1 - x0
        dy10 = y1 - y0
        dz10 = z1 - z0

        du = ux * dx10 + uy * dy10 + uz * dz10
        dv = vx * dx10 + vy * dy10 + vz * dz10

        return {
            'u': du + width / 2,
            'v': height / 2 - dv,
        }

    def map(self, u, v):
        fov = self.fov * np.pi / 180
        width = self.width
        height = self.height

        h0 = self.heading * np.pi / 180.0
        p0 = self.pitch * np.pi / 180.0

        f = 0.5 * width / np.tan(0.5 * fov)

        x0 = f * np.cos(p0) * np.sin(h0)
        y0 = f * np.cos(p0) * np.cos(h0)
        z0 = f * np.sin(p0)

        du = u - width / 2
        dv = height / 2 - v

        ux = sgn(np.cos(p0)) * np.cos(h0)
        uy = - sgn(np.cos(p0)) * np.sin(h0)
        uz = 0

        vx = -np.sin(p0) * np.sin(h0)
        vy = -np.sin(p0) * np.cos(h0)
        vz = np.cos(p0)

        x = x0 + du * ux + dv * vx
        y = y0 + du * uy + dv * vy
        z = z0 + du * uz + dv * vz

        r = np.sqrt(x * x + y * y + z * z)
        h = np.arctan2(x, y)
        p = np.arcsin(z / r)

        return {
            'heading': h * 180.0 / PI,
            'pitch': p * 180.0 / PI
        }


def get_center_hp(x_min, y_min, x_max, y_max, curr_heading, curr_pitch, fov):
    # x_list = np.array([x_min, x_max])
    # y_list = np.array([y_min, y_max])
    # x_idx = np.argsort(np.abs(320-x_list))
    # x_min, x_max = x_list[x_idx]
    # y_min, y_max = y_list[x_idx]
    # Calculate the center coordinates
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Initialize the PanoramaViewer with appropriate values
    viewer = PanoramaViewer(fov=fov, width=640, height=640, heading=curr_heading, pitch=curr_pitch)

    # Map the center, start, and end coordinates to heading and pitch
    top_pitch = viewer.map(center_x, y_max)['pitch']
    bot_pitch = viewer.map(center_x, y_min)['pitch']
    # start_pitch = viewer.map(x_min, y_min)['pitch']
    # end_pitch = viewer.map(x_min, y_max)['pitch']
    # start_heading = viewer.map(x_min, (y_min + y_max)/2)['heading']
    # end_heading = viewer.map(x_max, (y_min + y_max)/2)['heading']

    # Update the viewer's heading and pitch to the center point's heading and pitch
    # center_heading, center_pitch = average_heading_pitch(start_heading, start_pitch, end_heading,
    #                                                      end_pitch)

    center = viewer.map(center_x, center_y)
    # center_pitch, center_heading = center['pitch'], center['heading']
    viewer.heading = center['heading']
    viewer.pitch = (top_pitch + bot_pitch) / 2

    # Calculate the difference in pitch between the start and end points
    # pitch_diff = calculate_fov_for_bounding_box(heading_min, pitch_min, heading_max, pitch_max)

    # Update the viewer's field of view (fov)
    # viewer.fov = calculate_fov_for_bounding_box(start_hp['heading'], start_hp['pitch'], end_hp['heading'], end_hp['pitch'])
    viewer.fov = abs(top_pitch - bot_pitch) * 1.2
    # print(f"Heading: {viewer.heading}, Pitch: {viewer.pitch}, FOV: {viewer.fov}")

    return viewer.heading, viewer.pitch, viewer.fov