#test yolo prediction with its pretrained weights

from ultralytics import YOLO 
import os
import cv2 
import matplotlib.pyplot as plt

n_seg = "yolo_models\yolo11n-seg.pt"
s_seg = "yolo_models\yolo11s-seg.pt"
m_seg = "yolo_models\yolo11m-seg.pt"
l_seg = "yolo_models\yolo11l-seg.pt"

#yolov11-seg model variants
model_variant = [n_seg, s_seg, m_seg, l_seg]

model = YOLO(model_variant[0]) #pt loads offical pre-trained weights

# print(f"Loaded model with classes: {model.names}")

test_image_path = "dataset\BDD100K_inference_test\cabc30fc-e7726578.jpg"
image = cv2.imread(test_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image)

plt.figure(figsize=(10, 10))
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()