import cv2
import numpy as np
from nntools.dataset import nntools_wrapper


@nntools_wrapper
def fundus_autocrop(image: np.ndarray):
    r_img = image[:, :, 0]
    threshold = 10
    _, mask = cv2.threshold(r_img, threshold, 1, cv2.THRESH_BINARY)
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

@nntools_wrapper
def fundus_precise_autocrop(image:np.ndarray):
    h, w, c = image.shape
    img = image[:,:,0]
    threshold = 50
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    img = img.astype(np.uint8) * 255
    param1 = 15
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        2,
        min(h, w) // 8,
        param1=param1,
        param2=10,
        minRadius=int(min(h, w) // 4),
        maxRadius=int(max(h, w)//2),
    )
    if circles is None:
        return fundus_autocrop(image=image)
    mask = np.zeros_like(img)
    circle = np.uint16(np.around(circles))[0, 0]
    mask = (cv2.circle(mask, (circle[0], circle[1]), circle[2], 1, -1) > 0 ).astype(np.uint8)

    not_null_pixels = cv2.findNonZero(mask)
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {"image": image, "mask": mask}

    return {
        "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "mask": mask[y_range[0] : y_range[1], x_range[0] : x_range[1]],

    }
    
@nntools_wrapper
def image_check(image: np.ndarray):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return {"image": image}
    
    