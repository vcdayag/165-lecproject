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
capture = cv2.VideoCapture("data/VID_20231220_132423.mp4")
# capture = cv2.VideoCapture(0) # to use camera
assert capture.isOpened(), "Error reading video file"
w, h, fps = (
    int(capture.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
# REGION_POINTS = [(900, 0), (950, 0), (950, 1080), (900, 1080)]  # line or region points

# sets the region where detection will occur
LEFTREGION = w // 2 + 100
RIGHTREGION = LEFTREGION + 50
REGION_POINTS = [
    (LEFTREGION, 0),
    (RIGHTREGION, 0),
    (RIGHTREGION, 1080),
    (LEFTREGION, 1080),
]  

CLASSES_TO_COUNT = [2, 7]  # selects car and truck classes

# initialize the video writer
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# initialize the object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=False, reg_pts=REGION_POINTS, classes_names=model.names, draw_tracks=True
)

stickered_car_counter = 0
car_counter = 0
car_list = []

while True and capture.isOpened():
    success, input_img = capture.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # perform object tracking on the input image
    tracks = model.track(
        input_img, persist=True, show=False, classes=CLASSES_TO_COUNT, verbose=False
    )

    counted_image = counter.start_counting(input_img, tracks)

    # if no car was detected in the image
    if counted_image is None:
        cv2.imshow("MainImage", input_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # print(counter.in_counts)
    # print(counter.out_counts)

    # if a car/truck is detected
    if car_counter < counter.out_counts:
        car_counter += 1
        cv2.imwrite(f"output/car_{car_counter}.jpg", counted_image) # saves the image containing the car
        mask = StickerMask(input_img) # applies a red and yellow mask
        result = StickerResult(input_img, mask) 

        # print(tracks[0].boxes)
        car_list.append(tracks[0].boxes)

        possibleStickers = detectShape(result, mask, car_counter)
        if len(possibleStickers) > 0:
            stickered_car_counter += 1

        if True:
            print("Cars detected:", car_counter)
            print("Cars with stickers:", stickered_car_counter)
            print()
        # cv2.imshow("STICKER", result)
        # cv2.waitKey(0)

    # print(input_img)
    cv2.imshow("MainImage", counted_image)
    # create output file
    # video_writer.write(input_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"\nTotal number of cars: {car_counter}")
print(f"Total number of cars with stickers: {stickered_car_counter}")

capture.release()
video_writer.release()
cv2.destroyAllWindows()
