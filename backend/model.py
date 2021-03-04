"""
The code was modified accordingly by taking some code 
from https://github.com/harimkang/food-image-classifier.
"""

from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import time
import os

###### Detection
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow logging off

class ClassificationModel:
    """
    [Tensorflow Imagenet Model]
    # Class created using model provided in keras.applications
    """

    def __init__(self, class_list, img_width, img_height, batch_size=16) -> None:

        backend.clear_session()

        self.class_list = class_list
        self.img_width, self.img_height = img_width, img_height
        # batch_size can be up to 16 based on GPU 4GB (not available for 32)
        self.batch_size = batch_size

        self.model = None

        self.train_data = None
        self.validation_data = None
        self.num_train_data = None

        self.day_now = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        self.checkpointer = None
        self.csv_logger = None
        self.history = None
        self.model_name = "inception_v3"

    def generate_train_val_data(self, data_dir="dataset/train/"):
        """
        # Create an ImageDataGenerator by dividing the train and validation set
        # by 0.8/0.2 based on the train dataset folder.
        # train : 60600 imgs / validation : 15150 imgs
        """
        num_data = 0
        for root, dirs, files in os.walk(data_dir):
            if files:
                num_data += len(files)

        self.num_train_data = num_data
        _datagen = image.ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )
        self.train_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )
        self.validation_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )

    def set_model(self, model_name="inception_v3"):
        """
        # This is a function that composes a model, and proceeds to compile.
        # [Reference] - https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3
        """
        if model_name == "inception_v3":
            self.model = InceptionV3(weights="imagenet", include_top=False)
        elif model_name == "mobilenet_v2":
            self.model = MobileNetV2(weights="imagenet", include_top=False)
            self.model_name = model_name
        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        pred = Dense(
            len(self.class_list),
            kernel_regularizer=regularizers.l2(0.005),
            activation="softmax",
        )(x)

        self.model = Model(inputs=self.model.input, outputs=pred)
        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return 1

    def train(self, epochs=10):
        """
        # Training-related environment settings (log, checkpoint) and training
        """
        train_samples = self.num_train_data * 0.8
        val_samples = self.num_train_data * 0.2

        os.makedirs(os.path.join("models", "checkpoint"), exist_ok=True)
        self.checkpointer = ModelCheckpoint(
            filepath=f"models/checkpoint/{self.model_name}_checkpoint_{self.day_not}.hdf5",
            verbose=1,
            save_best_only=True,
        )
        os.makedirs(os.path.join("logs", "training"), exist_ok=True)
        self.csv_logger = CSVLogger(
            f"logs/training/{self.model_name}_history_model_{self.day_now}.log"
        )

        self.history = self.model.fit_generator(
            self.train_data,
            steps_per_epoch=train_samples // self.batch_size,
            validation_data=self.validation_data,
            validation_steps=val_samples // self.batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[self.csv_logger, self.checkpointer],
        )

        self.model.save(f"models/{self.model_name}_model_{self.day_now}.hdf5")

        return self.history

    def evaluation(self, batch_size=16, data_dir="test/", steps=5):
        """
        # Evaluate the model using the data in data_dir as a test set.
        """
        if self.model is not None:
            test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=batch_size,
                class_mode="categorical",
            )
            scores = self.model.evaluate_generator(test_generator, steps=steps)
            print("Evaluation data: {}".format(data_dir))
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        else:
            print("Model not found... : load_model or train plz")

    def prediction(self, image_data=None, img_path=None, show=False, save=False):
        """
        # Given a path for an image, the image is predicted and displayed through plt.
        """
        if img_path is not None:
            target_name = img_path.split(".")[0]
            target_name = target_name.split("/")[-1]
            save_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

            img = image.load_img(
                img_path, target_size=(self.img_height, self.img_width)
            )
            img = image.img_to_array(img)
        elif image_data is not None:
            img = image_data
            # img = np.asarray(img)
        else:
            raise Exception("image binary data or image_path needed...")

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        if self.model is not None:
            pred = self.model.predict(img)
            index = np.argmax(pred)
            self.class_list.sort()
            pred_value = self.class_list[index]
            if show:
                plt.imshow(img[0])
                plt.axis("off")
                plt.title("prediction: {}".format(pred_value))
                print("[Model Prediction] {}: {}".format(target_name, pred_value))
                plt.show()
                if save:
                    os.makedirs("results", exist_ok=True)
                    plt.savefig(
                        f"results/{self.model_name}_example_{target_name}_{save_time}.png"
                    )
            return pred_value
        else:
            print("Model not found... : load_model or train plz")
            return 0

    def load(self, model_path=None):
        """
        # If an already trained model exists, load it.
        """
        try:
            if model_path is None:
                model_path = "models/checkpoint/"
                os.makedirs(model_path, exist_ok=True)
                model_list = os.listdir(model_path)
                if model_list:
                    h5_list = [file for file in model_list if file.endswith(".hdf5")]
                    h5_list.sort()
                    backend.clear_session()
                    self.model = load_model(model_path + h5_list[-1], compile=False)
            else:
                backend.clear_session()
                self.model = load_model(model_path, compile=False)
                print("Model loaded...: ", model_path)

            self.model.compile(
                optimizer=SGD(lr=0.0001, momentum=0.9),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return 1
        except:
            print("Model not found... : train plz")
            return 0

    def show_accuracy(self):
        """
        # Shows the accuracy graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = f"model_accuracy_{self.day_now}"
            plt.title(title)
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train_acc", "val_acc"], loc="best")
            plt.show()
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/accuracy_{self.model_name}_model_{self.day_now}.png")
        else:
            print("Model not found... : load_model or train plz")

    def show_loss(self):
        """
        # Shows the loss graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = f"model_loss_{self.day_now}"
            plt.title(title)
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train_loss", "val_loss"], loc="best")
            plt.show()
            os.makedirs("results", exist_ok=True)
            plt.savefig(
                f"results/loss_{self.model_name}_model_{self.day_now}.png".format()
            )
        else:
            print("Model not found... : load_model or train plz")

class DetectorModel:
    def __init__(self, weights='inference_models/yolov5s_tomato_3classes.pt', source='test_images',
                 img_size=416, conf_thres=0.25, iou_thres=0.45, device='',
                 view_img=False, save_txt=False, save_conf=False, classes=None,
                 agnostic_nms=False, augment=False, update=False, project='./detect_results/',
                 name='tomato', exist_ok=True):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.classes=classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok

        # check_requirements()

        # print(self.update)
        # with torch.no_grad():
        #     if self.update:  # update all models (to fix SourceChangeWarning)
        #         for self.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #             self.detect()
        #             strip_optimizer(self.weights)
        #     else:
        #         self.detect()

    def detect(self):
        source, weights, view_img, save_txt, imgsz = self.source, self.weights, self.view_img, self.save_txt, self.img_size
        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None

        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')