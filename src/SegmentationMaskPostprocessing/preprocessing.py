import cv2
import numpy as np


def clean_mask(mask):
    """
    Clean the mask using morphological operations.

    :param mask: 2D numpy array, the mask
    :return: 2D numpy array, the cleaned mask
    """
    kernel = np.ones((5, 5), np.uint8)
    # Apply morphological operations
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned


def merge_mask(mask: np.ndarray, depth_map: np.ndarray, depth_threshold: float = 0.5):
    """
    Merge clusters in the mask based on depth similarity.

    :param mask: 2D numpy array, the mask
    :param depth_map: 2D numpy array, the depth map
    :param depth_threshold: float, the depth threshold for merging clusters
    :return: 2D numpy array, the merged mask
    """
    num_labels, labels_im = cv2.connectedComponents(mask)
    cluster_depths = []
    for label in range(1, num_labels):
        cluster_mask = (labels_im == label)
        cluster_depth = np.mean(depth_map[cluster_mask])
        cluster_depths.append((label, cluster_depth))
    # Merge clusters based on depth threshold
    merged_labels = {}
    for i, (label_i, depth_i) in enumerate(cluster_depths):
        for j, (label_j, depth_j) in enumerate(cluster_depths):
            if i < j and abs(depth_i - depth_j) < depth_threshold:
                merged_labels[label_j] = label_i
    # Update labels_im with merged labels
    for old_label, new_label in merged_labels.items():
        labels_im[labels_im == old_label] = new_label

    return labels_im


def depth_clean_mask(mask, depth_map, sidewalk_label=1, depth_threshold=0.25):
    """
    Remove the parts of the mask that have depth values significantly different from the rest of the sidewalk.

    :param mask: 2D numpy array, the mask
    :param depth_map: 2D numpy array, the depth map
    :param depth_threshold: float, the depth threshold for removing parts of the mask
    :param sidewalk_label: int, the label of the sidewalk in the mask
    :return: 2D numpy array, the cleaned mask
    """
    mean_depth = np.mean(depth_map[mask == sidewalk_label])
    mask_cleaned = mask.copy()
    mask_cleaned = np.where(np.abs(depth_map - mean_depth) > depth_threshold, 0, mask_cleaned)

    return mask_cleaned
