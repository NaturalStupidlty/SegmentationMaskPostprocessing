import plotly.express as px
import cv2

from loading import load_depth_map, load_mask
from preprocessing import clean_mask, merge_mask, depth_clean_mask
from postprocessing import compute_centroid, compute_sidewalk_width


def main():
    mask = load_mask("assets/240/IMG_0270.png")
    depth_map = load_depth_map("assets/240/depth_20241020_154356.bin")
    depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = clean_mask(mask)
    mask = merge_mask(mask, depth_map)
    mask = depth_clean_mask(mask, depth_map)

    centroid = compute_centroid(mask, depth_map, sidewalk_label=1, dbscan=False)
    sidewalk_width = compute_sidewalk_width(mask, depth_map, centroid)

    if sidewalk_width is not None:
        print(f"The estimated sidewalk width at the centroid is {sidewalk_width:.6f} meters.")
    else:
        print("Could not compute the sidewalk width.")

    # Visualization
    fig = px.imshow(depth_map, color_continuous_scale='Viridis', labels={'color': 'Depth'})
    fig.add_shape(type="circle",
                  x0=centroid[0] - 5, y0=centroid[1] - 5,
                  x1=centroid[0] + 5, y1=centroid[1] + 5,
                  line=dict(color="red", width=2))
    fig.update_layout(title="Depth Map Visualization", coloraxis_colorbar=dict(title="Depth"))
    fig.show()

    fig = px.imshow(mask, labels={'color': 'Mask'})
    fig.update_layout(title="Mask Visualization")
    fig.show()


if __name__ == "__main__":
    main()
