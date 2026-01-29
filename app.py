import streamlit as st
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Helmet Detection", layout="centered")

st.title("Helmet Detection using YOLOv8")
st.write("Upload an image or video to detect helmets.")

# Load trained model (PATH CHECK KAR LENA)
model = YOLO("weights/trained/best.pt")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    # ✅ IMPORTANT FIX: preserve file extension
    suffix = os.path.splitext(uploaded_file.name)[1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    if uploaded_file.type.startswith("image"):
        st.image(input_path, caption="Input Image", use_column_width=True)

        results = model.predict(
            source=input_path,
            conf=0.4,
            save=True
        )

        output_path = results[0].save_dir
        output_file = os.listdir(output_path)[0]

        st.image(
            os.path.join(output_path, output_file),
            caption="Detected Output",
            use_column_width=True
        )

    else:
        st.video(input_path)

        model.predict(
            source=input_path,
            conf=0.4,
            save=True
        )

        st.success("Video processed successfully. Check output folder.")
