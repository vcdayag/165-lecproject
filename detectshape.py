from stickerdetection import checkColor
import cv2

AREA_MIN = 2000
AREA_MAX = 5000000
MIN_YELLOW = 50
MIN_RED = 50


def detectShape(img: cv2.typing.MatLike, mask: cv2.typing.MatLike, car_number: int):
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from RGB to grayscale
    contour_mask = mask.copy() # saves a copy of the mask

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # find the contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    contours = contours[1:]

    contour_bounding_box = []

    # list for storing names of shapes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c) # extracting the coordinates
        contour_area = w * h

        # filters contours too small or too large
        if contour_area < AREA_MIN or contour_area > AREA_MAX:
            continue

        # store the bounding boxes of all the contours that pass the area filter
        contour_bounding_box.append((x, y, w, h))
        cv2.drawContours(contour_mask, [c], -1, (0, 0, 0), -1)

    # displaying the image after drawing contours
    final_mask = cv2.bitwise_xor(contour_mask, mask)
    final_image = cv2.bitwise_and(img, img, mask=final_mask)

    possibleStickers = []

    for c in contour_bounding_box:
        X, Y, W, H = c

        # extracts a specific region from the final image
        # that corresponds to a detected shape or contour
        croppedimage = final_image[Y : Y + H, X : X + W]

        # returns a binary mask where yellow pixels are white and the rest are black
        yellow_output = checkColor(croppedimage, "yellow2")
        yellow_pixels = cv2.countNonZero(yellow_output) # counts amount of yellow pixels

       

        # returns a binary mask where red pixels are white and the rest are black
        red_output = checkColor(croppedimage, "red1")
        red_pixels = cv2.countNonZero(red_output) # counts amount of red pixels

        # filters out contours that don't have enough red or yellow
        if yellow_pixels < MIN_YELLOW or red_pixels < MIN_RED:
            continue

        # bounds the detected sticker
        final_image = cv2.rectangle(
            final_image, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 255, 0), 2
        )
        possibleStickers.append(c)

        if True:
            print("   red:", red_pixels)
            print("yellow:", yellow_pixels)
            print()

        cv2.imwrite(
            f"output/car_{car_number}_{red_pixels}_{yellow_pixels}.jpg", croppedimage
        )

    return possibleStickers


if __name__ == "__main__":
    img = cv2.imread("data/IMG_20231220_132420.jpg")
    detectShape(img)
