import streamlit as st

from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend = "http://localhost:8000"

def classificate(image, server):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    resp = requests.post(
        server + "/classification",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )
    return resp

def yolo_detector(image, server):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})

    resp = requests.post(
        server + "/detector",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )
    return resp


def main():
    st.title("Plant Disease Detector Application")

    st.write(
        """Test Plant Image
            This streamlit example uses a FastAPI service as backend.
            Visit this URL at `:8000/docs` for FastAPI documentation.
            We provide 2 models and you can choose between 2 modes : classfier and detector.
            """
    )  # description and instructions

    # Side Bar
    st.sidebar.title("Test Models")
    app_mode = st.sidebar.selectbox("Choose Model", ["Mobilenet-v2", "yolov5s"])

    if app_mode == "Mobilenet-v2":
        print('model - mobilenet v2')
        run_app()
    elif app_mode == "yolov5s":
        print('model - yolov5s')
        run_app_detector()

def run_app():
    input_image = st.file_uploader("insert image")  # image upload widget

    if st.button("Detect Plant Disease"):

        col1, col2 = st.beta_columns(2)

        if input_image:
            pred = classificate(input_image, backend)
            original_image = Image.open(input_image).convert("RGB")
            predicted_value = pred.content
            col1.header("Original")
            col1.image(original_image, use_column_width=True)
            col2.header("Predicted")
            col2.write(str(predicted_value))

        else:
            # handle case with no image
            st.write("Insert an image!")

def run_app_detector():
    input_image = st.file_uploader("insert image")  # image upload widget

    if st.button("Detect Plant Disease"):
        col1, col2 = st.beta_columns(2)

        if input_image:
            pred = yolo_detector(input_image, backend)
            original_image = Image.open(input_image).convert("RGB")
            predicted_image = Image.open(pred.content).convert('RGB') # pred image
            col1.header("Original")
            col1.image(original_image, use_column_width=True)

            col2.header("Predicted")
            col2.image(predicted_image, use_column_width=True)

        else:
            # handle case with no image
            st.write("Insert an image!")

if __name__ == "__main__":
    main()