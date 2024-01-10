# https://docs.ultralytics.com/guides/object-counting/#real-world-applications
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from stickerdetection import StickerMask, StickerResult
from detectshape import detectShape
import os

if not os.path.exists("output/"):
    os.mkdir("output/")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/VID_20231220_132423.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# REGION_POINTS = [(900, 0), (950, 0), (950, 1080), (900, 1080)]  # line or region points
LEFTREGION = w // 2 + 100
RIGHTREGION = LEFTREGION + 50
REGION_POINTS = [
    (LEFTREGION, 0),
    (RIGHTREGION, 0),
    (RIGHTREGION, 1080),
    (LEFTREGION, 1080),
]  # line or region points
CLASSES_TO_COUNT = [2, 7]  # car and truck classes for count

# Video writer
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=False, reg_pts=REGION_POINTS, classes_names=model.names, draw_tracks=True
)

stickeredcarcounter = 0
carcounter = 0
trackedcars = []

while True and cap.isOpened():
    success, im0 = cap.read()

    tracks = model.track(
        im0, persist=True, show=False, classes=CLASSES_TO_COUNT, verbose=False
    )

    counted_image = counter.start_counting(im0, tracks)

    if counted_image is None:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    # print(counter.in_counts)
    # print(counter.out_counts)
    if carcounter < counter.out_counts:
        carcounter += 1

        cv2.imwrite(f"output/car_{carcounter}.jpg", counted_image)

        mask = StickerMask(im0)
        result = StickerResult(im0, mask)

        # print(tracks[0].boxes)
        trackedcars.append(tracks[0].boxes)

        possibleStickers = detectShape(result, mask, carcounter)
        if len(possibleStickers) > 0:
            stickeredcarcounter += 1

        if True:
            print("totalcars:", carcounter)
            print("stickered:", stickeredcarcounter)
            print()
        # cv2.imshow("STICKER", result)
        # cv2.waitKey(0)

    # print(im0)
    cv2.imshow("MainImage", counted_image)
    # create output file
    # video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(carcounter)
print(stickeredcarcounter)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
