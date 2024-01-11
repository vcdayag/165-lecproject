# https://docs.ultralytics.com/guides/object-counting/#real-world-applications
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from stickerdetection import StickerMask, StickerResult
from detectshape import detectShape
import os
import threading

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

DISPLAY_EXTERNAL_IMAGE = False


class ObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("UP Sticker Detection App")

        if not os.path.exists("output/"):
            os.mkdir("output/")

        self.model = YOLO("yolov8n.pt")
        self.video_source = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video files", "*.mp4")]
        )
        self.capture = cv2.VideoCapture(self.video_source)
        # capture = cv2.VideoCapture(0) # to use camera
        assert self.capture.isOpened(), "Error reading video file"
        self.w, self.h, self.fps = (
            int(self.capture.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )

        # Canvas for displaying the image
        self.canvas = tk.Canvas(
            self.master, width=int(self.w / 2), height=int(self.h / 2)
        )
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.total_cars_label = tk.Label(self.master, text="Total number of cars: 0")
        self.total_cars_label.grid(row=0, column=1, padx=10, pady=10)

        self.stickered_cars_label = tk.Label(
            self.master, text="Total number of cars with stickers: 0"
        )
        self.stickered_cars_label.grid(row=1, column=1, padx=10, pady=10)

        # Buttons and labels on the right
        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.grid(row=2, column=1, padx=10, pady=10)

        # Start object detection
        self.detection_thread = threading.Thread(target=self.start_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def start_detection(self):
        # REGION_POINTS = [(900, 0), (950, 0), (950, 1080), (900, 1080)]  # line or region points
        # sets the region where detection will occur
        LEFTREGION = self.w // 2 + 100
        RIGHTREGION = LEFTREGION + 50
        REGION_POINTS = [
            (LEFTREGION, 0),
            (RIGHTREGION, 0),
            (RIGHTREGION, 1080),
            (LEFTREGION, 1080),
        ]

        CLASSES_TO_COUNT = [2]  # selects car classes

        # initialize the video writer
        self.video_writer = cv2.VideoWriter(
            "object_counting_output.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.w, self.h),
        )

        # initialize the object Counter
        counter = object_counter.ObjectCounter()
        counter.set_args(
            view_img=False,
            reg_pts=REGION_POINTS,
            classes_names=self.model.names,
            draw_tracks=True,
        )

        stickered_car_counter = 0
        car_counter = 0
        car_list = []

        while True and self.capture.isOpened():
            success, input_img = self.capture.read()
            if not success:
                print(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            # perform object tracking on the input image
            tracks = self.model.track(
                input_img,
                persist=True,
                show=False,
                classes=CLASSES_TO_COUNT,
                verbose=False,
            )

            counted_image = counter.start_counting(input_img, tracks)

            # if no car was detected in the image
            if counted_image is None:
                if DISPLAY_EXTERNAL_IMAGE:
                    cv2.imshow("MainImage", input_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.quit()

                self.update_pics(input_img)
                continue

            # if a car is detected
            if car_counter < counter.out_counts:
                car_counter += 1
                cv2.imwrite(
                    f"output/car_{car_counter}.jpg", counted_image
                )  # saves the image containing the car
                mask = StickerMask(input_img)  # applies a red and yellow mask
                result = StickerResult(input_img, mask)

                car_list.append(tracks[0].boxes)

                possibleStickers = detectShape(result, mask, car_counter)
                if len(possibleStickers) > 0:
                    stickered_car_counter += 1

                if True:
                    print("Cars detected:", car_counter)
                    print("Cars with stickers:", stickered_car_counter)
                    print()
                    self.total_cars_label.config(
                        text=f"Total number of cars: {car_counter}"
                    )
                    self.stickered_cars_label.config(
                        text=f"Total number of cars with stickers: {stickered_car_counter}"
                    )

            self.update_pics(counted_image)
            if DISPLAY_EXTERNAL_IMAGE:
                cv2.imshow("MainImage", counted_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.quit()

            # self.master.update_idletasks()

        # print(f"\nTotal number of cars: {car_counter}")
        # print(f"Total number of cars with stickers: {stickered_car_counter}")

    def update_pics(self, pics):
        resized_image = cv2.resize(pics, (int(self.w / 2), int(self.h / 2)))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        image = ImageTk.PhotoImage(image)

        # Update canvas with the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def quit(self):
        self.capture.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
