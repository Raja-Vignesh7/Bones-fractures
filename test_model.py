import cv2
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import json
import shutil
# Load your trained YOLO model
model1 = YOLO("best (1).pt")  # Ensure the correct path to your trained model
model2 = YOLO("best.pt")
# Load the test image

def get_image_path():
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    image_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    filename = os.path.basename(image_path)

    filename = filename.replace(".png",".jpg.json")
    return image_path,filename
# print(f"base name: {os.path.basename(image_path)}")
def get_directory():
    root = tk.Tk()
    root.withdraw()

    # Open directory dialog
    directory_path = filedialog.askdirectory(title="Select a Directory")
    return directory_path

# # Print the selected directory
# print("Selected Directory:", directory_path)
def get_ann_path(directory_path,filename):
    for root, _, files in os.walk(directory_path):
            if filename in files:
                ann_path = os.path.join(root, filename)
                # print(ann_path)
                return ann_path
            else:
                return None

def read_ann(file_path):
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

def get_bbox(data):
    objects = data['objects']
    if objects:
        print(f'objects:-  {objects}')
        bbox = [obj['points']['exterior'] for obj in objects if obj['geometryType']=='rectangle']
        label = "fracture"
    else:
        bbox = [[0,0],[0,0]]
        label = 'not fracture'
        # print(bbox)
    # if objects:
    #             bbox = [
    #                     obj['points']['exterior'] for obj in objects if obj['geometryType'] == 'rectangle']
    #             bbox = bbox[0]
    return bbox[0],label

def get_original_image(image_path,bbox):
    image = cv2.imread(image_path)
    new_size = (544,544)
    print(image.shape)
    orig_h, orig_w = image.shape[:2]
    image = cv2.resize(image,(544,544))

    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    x1 = x1 / orig_w
    y1 = y1 / orig_h
    x2 = x2 / orig_w
    y2 = y2 / orig_h

    # Convert back to absolute coordinates for visualization
    x1 = int(x1 * new_size[0])
    y1 = int(y1 * new_size[1])
    x2 = int(x2 * new_size[0])
    y2 = int(y2 * new_size[1])

    cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
    return image
# image_path = "C:\\Users\\bvrvg\\Desktop\\Files\\ML\\DL\\Data sets\\BoneFractureYolo8\\train\\images\\fracture-of-the-humeral-capitellum-milch-type-1-1-1-_jpg.rf.8b022b79c884d0b37d923a3c961591c6.jpg"  # Replace with your actual image path

image_path,imagename = get_image_path()

directory_path = get_directory()

ann_path = get_ann_path(directory_path,imagename)

image = cv2.imread(image_path)




data = read_ann(ann_path)
bbox,org_label = get_bbox(data)

org_img = get_original_image(image_path,bbox)

# image = cv2.resize(image,(544,544),interpolation=cv2.INTER_AREA)
# Run YOLOv8 on the image
results1 = model1(image)
results2 = model2(image)
print(results1)
# Process detections
# Process detections from both models
for result in results1:
    for box in result.boxes:
        confidence = float(box.conf[0]) * 100  # Get confidence score in percentage
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        label = f"Fracture: {confidence:.2f}%"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for Model 1
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)

for result in results2:
    for box in result.boxes:
        confidence = float(box.conf[0]) * 100  # Get confidence score in percentage
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        label = f"Fracture: {confidence:.2f}%"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for Model 2
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 0), 2)
image = cv2.resize(image,(544,544),interpolation=cv2.INTER_AREA)
# Show the image with detections
cv2.imshow(f"original image {org_label}",org_img)
cv2.imshow("Fracture Detection", image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
