import cv2
import numpy as np
from nntools.dataset import nntools_wrapper


@nntools_wrapper
def fundus_autocrop(image: np.ndarray):
    r_img = image[:, :, 0]
    _, mask = cv2.threshold(r_img, 25, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(mask)
    mask = mask.astype(np.uint8)
    
    if not_null_pixels is None:
        return {"image": image, "mask": mask}
    
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {"image": image, "mask": mask}
    
    return {
        "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "mask": mask[y_range[0] : y_range[1], x_range[0] : x_range[1]],
    }
