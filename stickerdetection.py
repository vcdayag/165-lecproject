import numpy as np
import cv2

color_dict_HSV = {
    "black": [[180, 255, 30], [0, 0, 0]],
    "white": [[180, 18, 255], [0, 0, 231]],
    "red1": [[180, 255, 255], [159, 50, 70]],
    "red2": [[9, 255, 255], [0, 50, 70]],
    "green": [[89, 255, 255], [36, 50, 70]],
    "yellow": [[35, 255, 255], [25, 50, 70]],
    "yellow2": [[30, 255, 255], [20, 100, 100]],
    "blue": [[128, 255, 255], [90, 50, 70]],
    "purple": [[158, 255, 255], [129, 50, 70]],
    "orange": [[24, 255, 255], [10, 50, 70]],
    "gray": [[180, 18, 230], [0, 0, 40]],
}

kernel = np.ones((3, 3), np.uint8)
red_lower = np.array(color_dict_HSV["red1"][1])
red_upper = np.array(color_dict_HSV["red1"][0])
yellow_lower = np.array(color_dict_HSV["yellow2"][1])
yellow_upper = np.array(color_dict_HSV["yellow2"][0])


def checkColor(img: cv2.typing.MatLike, colortag: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        img,
        np.array(color_dict_HSV[colortag][1]),
        np.array(color_dict_HSV[colortag][0]),
    )
    return mask


def StickerMask(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    
    # original
    image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert from RGB to HSV

    red_mask = cv2.inRange(image, red_lower, red_upper) # create a binary mask identifying red regions
    red_mask = cv2.dilate(red_mask, kernel, iterations=2) # dilate to fill in gaps 

    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper) # create a binary mask identifying yellow regions
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2) # dilate to fill in gaps 

    mask = cv2.bitwise_or(yellow_mask, red_mask) # combines the red and yellow masks
    return mask


def StickerResult(
    img: cv2.typing.MatLike, mask: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    result = cv2.bitwise_and(img, img, mask=mask)
    return result
