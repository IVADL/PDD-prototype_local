from fastapi import FastAPI, File
from starlette.responses import Response

from classification import get_model

import io
from PIL import Image
import os
###
from model import DetectorModel

model = get_model("./inference_models/mobilenet_v2_checkpoint_202101281638.hdf5")
model_detector_path = "./inference_models/yolov5s_tomato_3classes.pt"

app = FastAPI(
    title="Plant Disease Detector",
    description="Plant Disease Detector using DL Models",
    version="0.1.0",
)

@app.post("/classification")
def get_predict_disease_result(file: bytes = File(...)):
    """"""
    image = Image.open(io.BytesIO(file)).convert("RGB")
    pred = model.prediction(image_data=image)

    return Response(content=pred, status_code=200)

@app.post("/detector")
def get_predict_disease_result_detector(file: bytes = File(...)):
    """"""
    test_img_path = './test_images/'
    result_img_path = './detect_results/'
    test_file = 'test_img.png'
    name_path = 'tomato/'

    image = Image.open(io.BytesIO(file)).convert("RGB")
    image.save(test_img_path+test_file)

    model_detector = DetectorModel(
        weights=model_detector_path,
        source=test_img_path,
        img_size=416,
        conf_thres=0.5, iou_thres=0.45, device='cpu', view_img=False, save_txt=False,
        save_conf=False, classes=None, agnostic_nms=False, augment=False, update=False,
        project = result_img_path, name=name_path, exist_ok=True
    )
    model_detector.detect()

    return Response(content='../backend/' + result_img_path + name_path + test_file, status_code=200)
