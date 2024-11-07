import asyncio

import pandas as pd
from ultralytics import YOLO

from utils.utils import download_images_from_df, sort_pano_locations, \
    fetch_metadata_for_pano, process_pano_metadata, select_panos, process_images_with_yolo

if __name__ == "__main__":
    # fetch meta-data from google using the location data and save to a csv file
    fetch_metadata_for_pano('data/resource_features_2.csv', 'data/cronulla_source_panoramas.csv')

    # filter too close or distance panos, too old panos and remove unnecessary columns from csv file
    df = pd.read_csv('data/cronulla_source_panoramas.csv')
    df = process_pano_metadata(df)
    df.to_csv('data/filtered_cronulla_panoramas.csv', index=False)

    # sort panoramas according to year and heading differences
    df = pd.read_csv('data/filtered_cronulla_panoramas.csv')
    df = sort_pano_locations(df)
    df.to_csv('data/sorted_cronulla_panoramas.csv', index=False)

    # select only 3 panoramas from each pole to download from the sorted list
    select_panos(3, 'data/sorted_cronulla_panoramas.csv', 'data/downloaded_cronulla_panoramas.csv')

    with asyncio.Runner() as runner:
        # download initial images asynchronously - make sure that the output directory is created
        runner.run(download_images_from_df('data/downloaded_cronulla_panoramas.csv', 'data/source_cronulla_images'))

        # download zoomed images and save calculated meta-data for images in a csv
        model = YOLO('models/bestv10.pt')
        processed_df = runner.run(
            process_images_with_yolo(model, 'data/downloaded_cronulla_panoramas.csv', 'data/source_cronulla_images',
                                     'data/zoomed_cronulla_images', 1000))
        processed_df.to_csv('data/zoomed_panoramas_cronulla.csv', index=False)

        # optional step to download images for pole material identification
        # runner.run(download_material_images(model, 'data/zoomed_panoramas_brighten.csv', 'data/steel-poles/source'))
