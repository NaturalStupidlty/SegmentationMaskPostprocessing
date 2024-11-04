import argparse

import cv2
import plotly.express as px

from loading import load_depth_map, load_mask
from postprocessing.imagery import (
    compute_centroid,
    compute_sidewalk_width,
)
from postprocessing.geospatial import get_location
from preprocessing.cleaning import clean_mask, depth_clean_mask
from preprocessing.merging import merge_mask


def parse_arguments():
    parser = argparse.ArgumentParser(description="Postprocess the segmentation mask.")
    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to the segmentation mask"
    )
    parser.add_argument(
        "--depth_map_path", type=str, required=True, help="Path to the depth map"
    )
    parser.add_argument(
        "--latitude", type=float, required=True, help="Latitude of the camera"
    )
    parser.add_argument(
        "--longitude", type=float, required=True, help="Longitude of the camera"
    )
    parser.add_argument("--yaw", type=float, required=True, help="Yaw of the camera")
    parser.add_argument(
        "--sidewalk_label",
        type=int,
        default=1,
        help="Label of the sidewalk in the mask",
    )
    parser.add_argument(
        "--dbscan", action="store_true", help="Use DBSCAN to compute the centroid"
    )
    return parser.parse_args()


def main():
    latitude = 50.3555513
    longitude = 30.4887025
    mask = load_mask("assets/240/IMG_0270.png")
    depth_map = load_depth_map("assets/240/depth_20241020_154356.bin")
    depth_map = cv2.resize(
        depth_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    mask = clean_mask(mask, kernel_size=5)
    mask = merge_mask(mask, depth_map)
    mask = depth_clean_mask(mask, depth_map)

    centroid = compute_centroid(mask, depth_map, sidewalk_label=1, dbscan=True)
    sidewalk_width = compute_sidewalk_width(mask, depth_map, centroid)
    sidewalk_latitude, sidewalk_longitude = get_location(
        depth_map,
        centroid,
        yaw=30,
        observer_latitude=latitude,
        observer_longitude=longitude,
    )

    print(f"Centroid: {centroid}")
    print(f"User: Latitude: {latitude}, Longitude: {longitude}")
    print(f"Sidewalk: Latitude: {sidewalk_latitude}, Longitude: {sidewalk_longitude}")
    print(f"Width: {sidewalk_width}")

    fig = px.imshow(
        depth_map, color_continuous_scale="Viridis", labels={"color": "Depth"}
    )
    fig.add_shape(
        type="circle",
        x0=centroid[0] - 5,
        y0=centroid[1] - 5,
        x1=centroid[0] + 5,
        y1=centroid[1] + 5,
        line=dict(color="red", width=2),
    )
    fig.update_layout(
        title="Depth Map Visualization", coloraxis_colorbar=dict(title="Depth")
    )
    fig.show()

    fig = px.imshow(mask, labels={"color": "Mask"})
    fig.update_layout(title="Mask Visualization")
    fig.show()


if __name__ == "__main__":
    main()
