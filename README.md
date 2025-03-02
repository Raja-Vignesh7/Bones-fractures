# X-Ray Fracture Detection System  

This project is a **fracture detection and classification system** using **YOLO models**. It allows users to upload X-ray images and detects possible fractures with bounding boxes and confidence scores.  

## 📂 Directory Structure  

```
📁 X-Ray Fracture Detection  
│── 📄 main.py              # Core logic for fracture detection using YOLO  
│── 📄 app.py               # Streamlit-based user interface  
│── 📁 saved_images         # Folder to store uploaded images and metadata  
│── 📄 images_metadata.json # JSON file storing detection results  
│── 📄 requirements.txt     # Dependencies  
│── 📄 README.md            # Project documentation  
│── 📄 best_new.pt          # YOLO model for fracture detection  
│── 📄 best (2).pt          # YOLO model for fracture classification  
│── 📁 model preformance    # model preformance
```  

---

## 🚀 Features  

✅ Upload X-ray images for fracture detection  
✅ Two YOLO models:  
   - **Model 1**: Detects if a fracture is present  
   - **Model 2**: Classifies the type of fracture  
✅ Draws bounding boxes and labels on detected fractures  
✅ Stores metadata of previous detections in `images_metadata.json`  
✅ Streamlit UI with zoom, brightness, and contrast adjustments  
✅ Ability to analyze new images or re-use previous detection results  

---

## 🛠 Installation  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/Raja-Vignesh7/Bones-fractures
cd X-Ray-Fracture-Detection
```  

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```  
### Note: python version-3.10

3️⃣ **Download YOLO models**  
Ensure `best_new.pt` and `best (2).pt` are present in the project directory.  

---

## 🎯 Usage  

### 🏗 Running the Streamlit App  

```bash
streamlit run app.py
```  

1️⃣ Upload an X-ray image  
2️⃣ The model will detect and classify fractures  
3️⃣ View results with bounding boxes and labels  
4️⃣ Adjust brightness, contrast, and zoom if needed  

---

## 🔍 How It Works  

### **1️⃣ Image Upload & Preprocessing**  
- Users upload an image via **Streamlit** (`app.py`).  
- The image is converted to a NumPy array for OpenCV processing.  

### **2️⃣ Fracture Detection (YOLO Model 1)**  
- The **first YOLO model (`best_new.pt`)** detects fractures.  
- Bounding boxes are drawn around detected areas.  

### **3️⃣ Fracture Classification (YOLO Model 2)**  
- The **second YOLO model (`best (2).pt`)** classifies the type of fracture.  
- Labels such as `elbow positive`, `humerus fracture`, etc., are assigned.  

### **4️⃣ Metadata Storage**  
- Detection results are stored in `images_metadata.json` for future reference.  
- If an image was previously processed, the app retrieves stored results.  

---

## 📌 Notes  

- **Try running detection multiple times** if results seem inconsistent.  
- **Previous results are stored** for quick access; reprocess if needed.  

---

## 👨‍💻 Contributors  

- **Raja Vignesh**  
- Open for contributions! 🚀  

---

This README provides a structured and user-friendly guide to understanding and running your **X-ray Fracture Detection System**. Let me know if you'd like any modifications! 🚀