import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
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

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self):
        iou_arr = []

        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.Detector()


        for img_path, annot_path in zip(self.images_paths, self.annot_paths):

            # Read an image
            img = cv2.imread(img_path)
            # cv2.imshow('win', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)

            # Read annotations:
            annot_list = self.get_annotations(annot_path)

            # Only for detection:
            p, gt = self.eval.prepare_for_detection(prediction_list, annot_list)

            iou = self.eval.iou_compute(p, gt)
            iou_arr.append(iou)

        miou = np.average(iou_arr)

        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")

        return miou

    def run_evaluation_vj_detector(self, detector):
        iou_arr = []

        for img_path, annot_path in zip(self.images_paths, self.annot_paths):
            img = cv2.imread(img_path)
            prediction_list = detector.detect(img)

            annot_list = self.get_annotations(annot_path)

            p, gt = self.eval.prepare_for_detection(prediction_list, annot_list)

            iou = self.eval.iou_compute(p, gt)
            iou_arr.append(iou)

        return np.mean(iou_arr)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()