import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from utils.utils import get_annotations
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

        self.images_paths = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        self.annot_paths = [
            os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt' for im_name in self.images_paths
        ]

        self.preprocess = Preprocess()
        self.eval = Evaluation()

    def run_evaluation(self):
        iou_arr = []

        # # Change the following detector and/or add your detectors below
        # import detectors.cascade_detector.detector as cascade_detector
        # # import detectors.your_super_detector.detector as super_detector
        # cascade_detector = cascade_detector.Detector()

        import detectors.yolov4_detector.detector as yolov4_detector
        yolov4_detector = yolov4_detector.Detector(0.5)

        for img_path, annot_path in zip(self.images_paths, self.annot_paths):

            # Read an image
            img = cv2.imread(img_path)
            # cv2.imshow('win', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = yolov4_detector.detect(img)

            # Read annotations:
            annot_list = get_annotations(annot_path)

            # Only for detection:
            p, gt = self.eval.prepare_for_detection(prediction_list, annot_list)

            iou = self.eval.iou_compute(p, gt)
            print(iou)
            iou_arr.append(iou)

        miou = np.average(iou_arr)

        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")
        print(miou)

        return miou

    def run_evaluation_vj(self):
        import detectors.cascade_detector.detector as cascade_detector
        cascade_detector = cascade_detector.Detector(scale_factor=1.015, min_neighbors=2)

        iou_arr = []

        for img_path, annot_path in zip(self.images_paths, self.annot_paths):
            img = cv2.imread(img_path)

            # print(img.shape)
            # Apply some preprocessing
            img = self.preprocess.blur(img, 'gaussian', (7,7))
            # cv2.imshow('win', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)

            # Read annotations:
            annot_list = get_annotations(annot_path)

            # Only for detection:
            p, gt = self.eval.prepare_for_detection(prediction_list, annot_list)

            iou = self.eval.iou_compute(p, gt)
            iou_arr.append(iou)
            print(np.average(iou_arr))

        miou = np.average(iou_arr)

        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")
        print(miou)

        return miou


    def run_evaluation_vj_analysis(self, detector, preprocess):
        iou_arr = []

        for img_path, annot_path in zip(self.images_paths, self.annot_paths):
            img = cv2.imread(img_path)
            img = preprocess(img)
            prediction_list = detector.detect(img)

            annot_list = get_annotations(annot_path)

            p, gt = self.eval.prepare_for_detection(prediction_list, annot_list)

            iou = self.eval.iou_compute(p, gt)
            iou_arr.append(iou)

        return np.mean(iou_arr)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()