import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import json
import torch

# Folder for saving images and metadata
SAVE_FOLDER = "saved_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# JSON file for metadata storage
METADATA_FILE = os.path.join(SAVE_FOLDER, "images_metadata.json")

# Load existing metadata if available
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as file:
        metadata = json.load(file)
else:
    metadata = {}


class Model:
    @staticmethod
    def load_metadata():
        if os.path.exists(METADATA_FILE) and os.path.getsize(METADATA_FILE) > 0:
            with open(METADATA_FILE, "r") as file:
                return json.load(file)
        else:
            return {}
            # print("Error: File is missing or empty.")
        # if os.path.exists(METADATA_FILE):
        #     with open(METADATA_FILE, "r") as file:
        #         return json.load(file)

    @staticmethod
    def save_metadata(metadata):
        with open(METADATA_FILE, "w") as file:
            json.dump(metadata, file, indent=4)

    @staticmethod
    def filter_subset_boxes(bbox_data):
        filtered_bboxes = []
        for i, bbox in enumerate(bbox_data):
            x1, y1, x2, y2 = bbox["coords"]
            is_subset = False
            for j, other_bbox in enumerate(bbox_data):
                if i != j:
                    ox1, oy1, ox2, oy2 = other_bbox["coords"]
                    if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:
                        is_subset = True
                        break
            if not is_subset:
                filtered_bboxes.append(bbox)
        return filtered_bboxes

    @staticmethod
    def predict(image, image_name):
        metadata = Model.load_metadata()
        if metadata and image_name in metadata  :
            bbox_data = metadata[image_name]  # Use existing data
        else:
            model1 = YOLO("best.pt")  # Detection model (fracture or not)
            model2 = YOLO("best_2.pt")  # Classification model (fracture type)

            results1 = model1(image)
            results2 = model2(image)

            bbox_data = []

            # Process results from model1 (Fracture Detection)
            for result in results1:
                for box in result.boxes:
                    class_id1 = int(box.cls)  # Fracture detection class
                    # print(f"Model1 label: {class_id1}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0]) * 100

                    bbox_data.append({
                        "coords": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id1,
                        "model": "model1"
                    })

            # Process results from model2 (Fracture Classification)
            for result in results2:
                # print(result.names)
                for box in result.boxes:
                    class_id2 = int(box.cls)  # Fracture classification class
                    # print(f"Model2 label: {class_id2}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0]) * 100

                    bbox_data.append({
                        "coords": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id2,
                        "model": "model2"
                    })

            bbox_data = Model.filter_subset_boxes(bbox_data)
            metadata[image_name] = bbox_data
            Model.save_metadata(metadata)
        fracture_labels = {0: 'elbow positive', 1: 'fingers positive',
                           2: 'forearm fracture', 3: 'humerus fracture', 
                           4: 'humerus', 5: 'shoulder fracture', 6: 'wrist positive'}

        # Draw bounding boxes and text on image
        for bbox in bbox_data:
            x1, y1, x2, y2 = bbox["coords"]
            confidence = bbox["confidence"]
            model = bbox["model"]
            class_id = bbox["class_id"]

            # Set color and label
            color = (0, 0, 255) if model == "model1" else (255, 0, 0)
            label = "Fracture" if model == "model1" else fracture_labels.get(class_id, "Unknown")

            # Adjust font size dynamically based on image size
            font_scale = max(0.5, min(image.shape[1] / 800, 1))
            thickness = max(1, int(font_scale * 2))

            # Move text inside the bounding box if it's too close to the edge
            text_x = x1
            text_y = y1 + 20 if y1 < 20 else y1 - 10  # Move text down if near top

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label}: {confidence:.2f}%", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        return image


def load_and_store_image():
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])

    if file_path:
        file_name = os.path.basename(file_path)
        save_path = os.path.join(SAVE_FOLDER, file_name)

        # Copy image to the save folder
        shutil.copy(file_path, save_path)

        # Read image
        image = cv2.imread(save_path)
        image = cv2.resize(image, (544, 544), interpolation=cv2.INTER_AREA)
        print(f"Image saved to: {save_path}")

        try:
            os.remove(save_path)
            print(f"Image '{save_path}' deleted successfully.")
        except FileNotFoundError:
            print(f"Error: Image '{save_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return image, file_name
    else:
        print("No image selected.")
        return None, None


# # Load image and process
# image, image_name = load_and_store_image()
# if image is not None:
#     img_res = Model.predict(image, image_name)
#     cv2.imshow("Fracture Detection", img_res)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No image selected.")
