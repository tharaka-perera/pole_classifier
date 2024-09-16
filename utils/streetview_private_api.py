import asyncio
from io import BytesIO
from typing import Dict, Union, List
import pandas as pd

import httpx
import requests
from PIL import Image

from utils.utils import save_image_async

async_client = httpx.AsyncClient()

DEFAULT_MAX_RETRIES = 6


class SVPrivateClient:
    def __init__(self, max_concurrent_requests: int):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    def get_streetview(
        self,
        pano_id: str,
        width: int = 640,
        height: int = 640,
        heading: float = 0,
        fov: int = 120,
        pitch: float = 0,
    ) -> Image.Image:
        """
        Get an image using the official API. These are not panoramas.

        You can find instructions to obtain an API key here:
        https://developers.google.com/maps/documentation/streetview/

        Args:
            pano_id (str): The panorama id.
            heading (int): The heading of the photo. Each photo is taken with a 360
                camera. You need to specify a direction in degrees as the photo
                will only cover a partial region of the panorama. The recommended
                headings to use are 0, 90, 180, or 270.
            width (int): Image width (max 640 for non-premium downloads).
            height (float): Image height (max 640 for non-premium downloads).
            fov (int): Image field-of-view.
            pitch (float): Image pitch.
        """

        url = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
        params: Dict[str, Union[str, int]] = {
            "w": width,
            "h": height,
            "cb_client": "search.gws-prod.gps",
            "thumbfov": fov,
            "pitch": -pitch,
            "yaw": heading,
            "panoid": pano_id,
        }

        response = requests.get(url, params=params, stream=True)
        img = Image.open(BytesIO(response.content))
        return img

    async def get_streetview_async(
        self,
        pano_id: str,
        width: int = 640,
        height: int = 640,
        heading: float = 0,
        fov: float = 120,
        pitch: float = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Image.Image:
        """
        Get an image using the official API. These are not panoramas.

        You can find instructions to obtain an API key here:
        https://developers.google.com/maps/documentation/streetview/

        Args:
            pano_id (str): The panorama id.
            heading (int): The heading of the photo. Each photo is taken with a 360
                camera. You need to specify a direction in degrees as the photo
                will only cover a partial region of the panorama. The recommended
                headings to use are 0, 90, 180, or 270.
            width (int): Image width (max 640 for non-premium downloads).
            height (float): Image height (max 640 for non-premium downloads).
            fov (int): Image field-of-view.
            pitch (float): Image pitch.
            max_retries (int): Maximum number of retries in case of a request error.
        """

        url = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
        params: Dict[str, Union[str, int]] = {
            "w": width,
            "h": height,
            "cb_client": "search.gws-prod.gps",
            "thumbfov": fov,
            "pitch": -pitch,
            "yaw": heading,
            "panoid": pano_id,
        }

        async with self.semaphore:
            for _ in range(max_retries):
                try:
                    response = await async_client.get(url, params=params)
                    return Image.open(BytesIO(response.content))

                except httpx.RequestError as e:
                    print(f"Request error {e}. Trying again in 2 seconds.")
                    await asyncio.sleep(2)

            raise httpx.RequestError("Max retries exceeded.")

    async def download_and_save_image(self, pano_id: str, heading: float, pitch: float, fov: float, filename: str):
        image = await self.get_streetview_async(pano_id, heading=heading, pitch=pitch, fov=fov)
        await save_image_async(image, filename)

    async def download_and_save_images(self, dataframe, output_dir):
        tasks = []
        for idx, row in dataframe.iterrows():
            index = row['index']
            pano_id = row['pano_id']
            year = row['year']
            distance = round(row['distance'])
            heading = row['heading']
            pitch = row['pitch']
            fov = row['fov']
            filename = f"{output_dir}/{index}_id_{idx}_d_{distance}_{pano_id}_y_{year}.jpg"
            task = self.download_and_save_image(pano_id, heading, pitch, fov, filename)
            tasks.append(task)

        await asyncio.gather(*tasks)
