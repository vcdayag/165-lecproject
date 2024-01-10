# https://docs.ultralytics.com/guides/object-counting/#real-world-applications
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from stickerdetection import StickerMask, StickerResult
from detectshape import detectShape

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/VID_20231220_132423.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

region_points = [(750, 0), (800, 0), (800, 1080), (750, 1080)]  # line or region points
classes_to_count = [2, 7]  # car and truck classes for count

# Video writer
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=False, reg_pts=region_points, classes_names=model.names, draw_tracks=True
)

carcounter = 0
trackedcars = []
while True and cap.isOpened():
    success, im0 = cap.read()

    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

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

        mask = StickerMask(im0)
        result = StickerResult(im0, mask)

        print(tracks[0].boxes)
        trackedcars.append(tracks[0].boxes)

        detectShape(result, mask)
        # cv2.imshow("STICKER", result)
        # cv2.waitKey(0)

    # print(im0)
    cv2.imshow("ultralytics", counted_image)
    # create output file
    # video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(trackedcars)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
