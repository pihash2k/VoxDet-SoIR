import numpy as np
import cv2

import numpy as np
import cv2


def mask_to_keypoints_ver2(mask, num_points=20):
    """
    Convert a binary mask to keypoints along the contour.

    Args:
    mask (numpy.ndarray): Binary mask (2D numpy array)
    num_points (int): Desired number of keypoints to generate

    Returns:
    list: List of (x, y) keypoint coordinates along the contour
    """
    # Ensure mask is binary and of type uint8
    mask = (mask > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return []

    # Use the longest contour
    contour = max(contours, key=cv2.contourArea)

    # Get the total length of the contour
    contour_length = cv2.arcLength(contour, True)

    # Calculate the distance between points
    point_distance = contour_length / num_points

    # Initialize variables
    keypoints = []
    accumulated_distance = 0
    prev_point = contour[0][0]

    # Iterate through the contour points
    for point in contour:
        point = point[0]
        distance = np.linalg.norm(point - prev_point)
        accumulated_distance += distance

        # If we've traveled far enough, add a keypoint
        while accumulated_distance >= point_distance:
            # Interpolate to get the exact point
            t = (point_distance - (accumulated_distance - distance)) / distance
            keypoint = prev_point * (1 - t) + point * t
            keypoints.append(tuple(map(int, keypoint)))

            accumulated_distance -= point_distance

        prev_point = point

    # Ensure we have exactly num_points
    if len(keypoints) > num_points:
        keypoints = keypoints[:num_points]
    elif len(keypoints) < num_points:
        # If we have too few points, duplicate the last one
        keypoints.extend([keypoints[-1]] * (num_points - len(keypoints)))

    return keypoints


def mask_to_keypoints(mask, num_points=20):
    """
    Convert a binary mask to keypoints along the contour.

    Args:
    mask (numpy.ndarray): Binary mask (2D numpy array)
    num_points (int): Approximate number of keypoints to generate

    Returns:
    list: List of (x, y) keypoint coordinates along the contour
    """
    # Ensure mask is binary and of type uint8
    mask = (mask > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Use the longest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify the contour to approximately num_points
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Adjust epsilon to get closer to num_points
    while len(approx) > num_points:
        epsilon *= 1.1
        approx = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to list of tuples in (x, y) format
    keypoints = [tuple(point[0]) for point in approx]

    return keypoints


def keypoints_to_mask(keypoints, shape):
    """
    Reconstruct an approximate mask from keypoints by creating a filled polygon.

    Args:
    keypoints (list): List of (x, y) keypoint coordinates
    shape (tuple): Shape of the original mask (height, width)

    Returns:
    numpy.ndarray: Reconstructed binary mask
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if len(keypoints) > 2:
        # Convert keypoints to the format expected by cv2.fillPoly
        pts = np.array(keypoints, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill the polygon
        cv2.fillPoly(mask, [pts], 1)

    return mask


def encode_rle(mask):
    """
    Encode a binary mask into RLE format.

    :param mask: 2D numpy array of shape (height, width) with values 0 or 1
    :return: Dictionary with 'counts' key containing the RLE encoded list and 'size' key with (height, width)
    """
    # Get the mask dimensions
    height, width = mask.shape

    # Flatten the mask
    mask = mask.flatten()

    # Find where the mask changes from 0 to 1 or 1 to 0
    diff = np.diff(mask)
    change_points = np.where(diff != 0)[0] + 1

    # Add start and end points
    change_points = np.concatenate(([0], change_points, [len(mask)]))

    # Calculate run lengths
    rle_counts = np.diff(change_points)

    # Convert to list
    rle_counts = rle_counts.tolist()

    return {"counts": rle_counts, "size": (height, width), "transpose": False}


def decode_rle(seg_dict, transpose=True):
    rle_counts = seg_dict["counts"]
    height, width = seg_dict["size"]
    if "transpose" in seg_dict:
        transpose = seg_dict["transpose"]
    mask = np.zeros(height * width, dtype=np.uint8)
    pixel = 0
    for i, count in enumerate(rle_counts):
        mask[pixel : pixel + count] = i % 2
        pixel += count
    if transpose:
        return mask.reshape(height, width).T
    else:
        return mask.reshape(height, width)


def decode_rle_list(masks):
    for ind, m in enumerate(masks):
        if isinstance(m, dict):
            m = decode_rle(m)
            masks[ind] = m
    return masks


# Example usage:
# mask = np.zeros((100, 100), dtype=np.uint8)
# mask[25:75, 25:75] = 1  # Create a square in the middle
# keypoints = mask_to_keypoints(mask)
