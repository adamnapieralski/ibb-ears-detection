import numpy as np
from yolov4 import Detector as DetectorYolov4
import cv2
from PIL import Image


class Detector:
    def __init__(self):
        self.det = DetectorYolov4(
            lib_darknet_path='/Users/adamnapieralski/Projects/private/erasmus/image-based-biometry/task_2/yolov4/libdarknet.so',
            config_path='/Users/adamnapieralski/Projects/private/erasmus/image-based-biometry/task_2/yolov4/cfg/yolov4-ears.cfg',
            weights_path='/Users/adamnapieralski/Projects/private/erasmus/image-based-biometry/task_2/yolov4/weights/yolov4-ears-1.weights',
            meta_path='/Users/adamnapieralski/Projects/private/erasmus/image-based-biometry/task_2/yolov4/cfg/ears.data'
        )
        self.net_shape = (self.det.network_width(), self.det.network_height())

    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_org_shape = img.shape[1], img.shape[0] # (w,h)

        img_pil = Image.fromarray(img)

        img_arr = np.array(img_pil.resize(self.net_shape))

        detections = self.det.perform_detect(image_path_or_buf=img_arr, thresh=0.05, show_image=False)

        scale_w, scale_h = img_org_shape[0] / self.net_shape[0], img_org_shape[1] / self.net_shape[1]

        return [np.array([d.left_x * scale_w, d.top_y * scale_h, d.width * scale_w, d.height * scale_h], dtype=np.int32) for d in detections]