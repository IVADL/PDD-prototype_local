

## PDD(Plant Diseases Detection) Prototype
- Dataset: https://www.aihub.or.kr/aidata/129
- Model
    - Classification & Detection
    - Yolov5 Github : https://github.com/ultralytics/yolov5
    
- Tool
    - Frontend : Streamlit - https://docs.streamlit.io/en/stable/#
    - Backend : FastAPI - https://fastapi.tiangolo.com/
    - Annotation Tool : CVAT - https://github.com/openvinotoolkit/cvat
    
## Directory Structure

```
┌─PDD(Plant Diseases Detection)
│
├─backend
│  ├─ models  # yolo v5 module
│  ├─ utils  # yolo v5 module
│  ├─ inference_models  # mobilenet_v2, yolov5s
│  ├─ util  # for classification label
│  ├─ test_images  # image upload -> save path
│  ├─ detect_results  # inference result image with bounding box
│  │
│  ├─ classification.py  # classfication 
│  ├─ server.py  # FastAPI Server
│  └─ model.py  # classficiation , detection model
│
├─frontend 
│  └─ main.py  # Streamlit Frontend
│
├────────────────────
├─requirements.txt
├─UI_테스트용_이미지_크기_1824
│
└────────────────────
```

<hr>

## Environment

### Dependencies
- plant-disease-detection was developed using the following library version:

- [Python3] - 3.8.5
- [Tensorflow] - 2.3.0
- [CUDA] - 10.1
- [Cudnn] - 7.6.5
- [Pytorch] - 1.7.1
- [Streamlit] - 0.77.0
- [FastAPI] - 0.60.1

### Env Setting

1. Clone food-image-classifier Repository

   ```sh
   $ git clone https://github.com/IVADL/PDD-prototype_local.git

    ```

2. Create conda env

   ```sh
   $ conda create -n pdd 
   $ conda activate pdd
   $ pip install -r requirements.txt

    ```
    
3. Run Backend
   ```sh
   $ cd backend
   $ uvicorn server:app --host "0.0.0.0" --port "8000"

    ```

4. Run Frontend
   ```sh
   $ cd frontend
   $ streamlit run main.py
   $ http://localhost:8501/

    ```
    
5. Run model
- Select model : mobilenet or yolov5
- Test Image upload
- Click 'Detect Plant Disease Button'
