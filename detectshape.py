import cv2
from stickerdetection import checkColor

AREA_MIN = 2000
AREA_MAX = 5000
MIN_YELLOW = 10
MIN_RED = 10


def detectShape(img: cv2.typing.MatLike, mask: cv2.typing.MatLike):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contourmask = mask.copy()

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    contours = contours[1:]

    contourBoundingBox = []

    # list for storing names of shapes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contourarea = w * h

        if contourarea < AREA_MIN or contourarea > AREA_MAX:
            continue

        contourBoundingBox.append((x, y, w, h))
        cv2.drawContours(contourmask, [contour], -1, (0, 0, 0), -1)

    # displaying the image after drawing contours
    finalmask = cv2.bitwise_xor(contourmask, mask)
    finalimage = cv2.bitwise_and(img, img, mask=finalmask)

    possibleStickers = []

    for c in contourBoundingBox:
        X, Y, W, H = c

        croppedimage = finalimage[Y : Y + H, X : X + W]
        red_output = checkColor(croppedimage, "red1")
        red_pixels = cv2.countNonZero(red_output)

        yellow_output = checkColor(croppedimage, "yellow2")
        yellow_pixels = cv2.countNonZero(yellow_output)

        if False:
            print("   red:", red_pixels)
            print("yellow:", yellow_pixels)
            print()

        if red_pixels > MIN_RED and yellow_pixels > MIN_YELLOW:
            finalimage = cv2.rectangle(
                finalimage, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 255, 0), 2
            )
            possibleStickers.append(c)

    # cv2.imshow("shapes", finalimage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return possibleStickers


if __name__ == "__main__":
    img = cv2.imread("data/IMG_20231220_132420.jpg")
    detectShape(img)
