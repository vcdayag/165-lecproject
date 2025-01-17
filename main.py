from ultralytics import YOLO
import numpy as np
import cv2

color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'yellow2': [[30, 255, 255], [20, 100, 100]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

# Load a pretrained YOLO model (recommended for training)

model = YOLO('yolov8n.onnx', task='detect') #CPU
# model = YOLO('yolov8n.pt') #GPU

# Perform object detection on an image using the model
results = model('data/VID_20231220_132423.mp4', stream=True)

kernel = np.ones((3,3), np.uint8)
red_lower = np.array(color_dict_HSV["red1"][1])
red_upper = np.array(color_dict_HSV["red1"][0])
yellow_lower = np.array(color_dict_HSV["yellow2"][1])
yellow_upper = np.array(color_dict_HSV["yellow2"][0])

for r in results:
    # original
    image = r.orig_img
    # image = r.plot() # with box
    result = image.copy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(image, red_lower, red_upper)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)    

    # dilate yellow mask
    # yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
    
    mask = cv2.bitwise_xor(yellow_mask,red_mask)
    result = cv2.bitwise_and(result, result, mask=mask)
    
    cv2.imshow('result', result)
    key = cv2.waitKey(1) & 0xff
    if key==ord('q'):
        break

cv2.destroyAllWindows()

# model.export(format='onnx')