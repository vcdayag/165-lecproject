from stickerdetection import checkColor
import cv2

AREA_MIN = 2000
AREA_MAX = 5000000
MIN_YELLOW = 50
MIN_RED = 50
PIXELS_AREA_RATIO = 0.30
RED_YELLOW_RATIO_MIN = 0.10
RED_YELLOW_RATIO_MAX = 5


def detectShape(img: cv2.typing.MatLike, mask: cv2.typing.MatLike, car_number: int):
    contour_mask = mask.copy()  # saves a copy of the mask

    # find the contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    contours = contours[1:]

    contour_bounding_box = []

    # list for storing names of shapes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)  # extracting the coordinates
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
        yellow_pixels = cv2.countNonZero(
            yellow_output
        )  # counts amount of yellow pixels

        # returns a binary mask where red pixels are white and the rest are black
        red_output = checkColor(croppedimage, "red1")
        red_pixels = cv2.countNonZero(red_output)  # counts amount of red pixels

        # filters out contours that don't have enough red or yellow
        if yellow_pixels < MIN_YELLOW or red_pixels < MIN_RED:
            continue

        totalpixels = red_pixels + yellow_pixels
        area = W * H
        red_yellow_ratio = yellow_pixels / red_pixels

        if False:
            print("        red:", red_pixels)
            print("     yellow:", yellow_pixels)
            print("totalpixels:", totalpixels)
            print("       area:", area)
            print("pix-A_ratio:", totalpixels / area)
            print("   RY_ratio:", red_yellow_ratio)
            print()
            cv2.imwrite(
                f"output/debug_car_{car_number}_{red_pixels}_{yellow_pixels}.jpg",
                croppedimage,
            )

        # check if the number of pixel is less than 45% of area
        if totalpixels < area * PIXELS_AREA_RATIO:
            continue

        # check if the red and yellow pixel ration
        if (
            red_yellow_ratio < RED_YELLOW_RATIO_MIN
            or red_yellow_ratio > RED_YELLOW_RATIO_MAX
        ):
            continue

        # bounds the detected sticker
        final_image = cv2.rectangle(
            final_image, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 255, 0), 2
        )
        possibleStickers.append((X, Y, W, H, croppedimage))

        cv2.imwrite(
            f"output/car_{car_number}_{red_pixels}_{yellow_pixels}.jpg", croppedimage
        )

    return possibleStickers


if __name__ == "__main__":
    img = cv2.imread("data/IMG_20231220_132420.jpg")
    detectShape(img)
