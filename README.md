🦺 Helmet Detection using YOLOv8

This project implements a helmet detection system using a custom-trained YOLOv8 object detection model.
It includes a Streamlit web application for image and video inference, making the model easy to test and deploy.

🚀 Features

-Helmet detection using YOLOv8
-Custom trained model on a labeled helmet dataset
-Supports image and video inference
-Interactive Streamlit UI
-Clean local + Colab training workflow
-Deployment-ready structure

🧠 Model Details

Model: YOLOv8 (Ultralytics)

Task: Object Detection

Classes: Helmet

Training:

Performed on Google Colab (GPU)
Final trained weights saved as best.pt
Evaluation:
High precision and recall
Stable mAP indicating good generalization

📁 Project Structure
PPE-Detection-YOLO/
│
├── app.py                  # Streamlit application
├── src/                    # Inference scripts
│   ├── infer_pretrained_image.py
│   ├── infer_trained_image.py
│   ├── infer_pretrained_video.py
│   └── infer_trained_video.py
│
├── data/                   # Input images/videos (ignored in git)
├── weights/                # Trained model weights (ignored in git)
│   └── best.pt
│
├── outputs/                # Inference results (ignored in git)
├── notebooks/              # Training notebook (Colab)
├── requirements.txt
└── README.md

🛠 Installation (Local)

Create and activate virtual environment:

python -m venv yolo_venv
yolo_venv\Scripts\activate


Install dependencies:

pip install ultralytics streamlit opencv-python numpy matplotlib

▶️ Run Streamlit App
streamlit run app.py


Then open in browser:

http://localhost:8501

🖼️ Image Inference (Example)

Upload an image (.jpg, .png)

App detects helmets and displays bounding boxes with confidence scores

🎥 Video Inference (Example)

Upload a video (.mp4)

Helmet detection runs frame-by-frame

Output video is saved locally

📌 Notes

Dataset and trained weights are intentionally excluded from GitHub

Model weights can be retrained or replaced easily

Project can be extended to:

Full PPE detection

Webcam / CCTV (RTSP) feed

Cloud deployment

🔮 Future Improvements

Add helmet violation logic (person without helmet)

Integrate live webcam detection

Deploy on Streamlit Cloud / AWS

Extend to vest and safety shoes detection

👤 Author

Aman Deep
AI / Machine Learning Enthusiast
Focus: Computer Vision, YOLO, Real-world AI deployment