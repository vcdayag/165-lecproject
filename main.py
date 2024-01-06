from ultralytics import YOLO
import numpy as np
import cv2

# Load a pretrained YOLO model (recommended for training)

model = YOLO('yolov8n.onnx', task='detect') #CPU
# model = YOLO('yolov8n.pt') #GPU

# Perform object detection on an image using the model
results = model('data/VID_20231220_132423.mp4', stream=True)

for r in results:
    # original
    # image = r.orig_img
    image = r.plot() # with box
    result = image.copy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lower = np.array([155,25,0])
    red_upper = np.array([179,255,255])
    red_mask = cv2.inRange(image, red_lower, red_upper)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yellow_lower = np.array([200,200,0])
    yellow_upper = np.array([255,255,100])
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)
    
    mask = cv2.bitwise_xor(yellow_mask,red_mask)
    result = cv2.bitwise_and(result, result, mask=mask)
    
    cv2.imshow('result', result)
    key = cv2.waitKey(1) & 0xff
    if key==ord('q'):
        break

cv2.destroyAllWindows()

# model.export(format='onnx')