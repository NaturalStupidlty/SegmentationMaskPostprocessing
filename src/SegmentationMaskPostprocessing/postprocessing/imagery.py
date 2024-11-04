import numpy as np


def compute_centroid(mask, depth_map, sidewalk_label=1, dbscan=False):
    """
    Compute the centroid of the sidewalk mask.

    :param mask: 2D numpy array, the sidewalk mask
    :param depth_map: 2D numpy array, depth values in meters
    :param sidewalk_label: int, the label of the sidewalk in the mask
    :param dbscan: bool, whether to use DBSCAN for clustering
    :return: tuple (X, Y) of the centroid location in pixels
    """

    assert mask.shape == depth_map.shape

    indices = np.where(mask == sidewalk_label)
    X = indices[1]
    Y = indices[0]
    Z = depth_map[Y, X]

    if dbscan:
        from sklearn.cluster import DBSCAN

        points = np.stack((X, Y, Z), axis=-1)
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
        labels = clustering.labels_

        # Select points belonging to the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        inlier_points = points[labels == largest_cluster_label]

        # Compute centroid of inliers
        X_c, Y_c, Z_c = np.mean(inlier_points, axis=0)
        centroid = (X_c, Y_c)
    else:
        centroid = (np.median(X), np.median(Y))

    return centroid


def compute_sidewalk_width(mask, depth_map, centroid):
    """
    Compute the physical width of the sidewalk at the centroid location.

    :param mask: 2D numpy array, the sidewalk mask
    :param depth_map: 2D numpy array, depth values in meters
    :param centroid: tuple (X, Y) of the centroid location in pixels
    :return: physical width in meters
    """
    centroid_x, centroid_y = int(centroid[0]), int(centroid[1])

    # Find the leftmost and rightmost sidewalk pixels
    sidewalk_row = mask[centroid_y, :]
    sidewalk_indices = np.where(sidewalk_row > 0)[0]

    if len(sidewalk_indices) < 2:
        print("Cannot find sidewalk edges at the centroid location")
        return 0

    side = depth_map[centroid_y, centroid_x]

    left_pixel = sidewalk_indices[0]
    right_pixel = sidewalk_indices[-1]
    left_estimate = np.sqrt(depth_map[centroid_y, left_pixel] ** 2 - side**2)
    right_estimate = np.sqrt(depth_map[centroid_y, right_pixel] ** 2 - side**2)

    width = (right_estimate + left_estimate) / 2

    return width
