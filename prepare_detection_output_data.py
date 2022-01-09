import os
import glob
import cv2
import itertools
import shutil
import numpy as np
from pathlib import Path
from utils.utils import get_annotations
from detectors.yolov4_detector.detector import Detector
from metrics.evaluation import Evaluation

D_TYPE = 'test'

detector = Detector(det_type=1)
evaluator = Evaluation()

images_paths = sorted(glob.glob('data/ears/' + D_TYPE + '/*.png', recursive=True))
annot_paths = [
    os.path.join(f'data/ears/annotations/detection/{D_TYPE}_YOLO_format', Path(os.path.basename(im_name)).stem) + '.txt' for im_name in images_paths
]

for img_path, annot_path in zip(images_paths, annot_paths):
    img = cv2.imread(img_path)
    prediction_list = detector.detect(img)
    annot_list = get_annotations(annot_path)

    if len(annot_list) != 0:
        [x, y, w, h] = annot_list[0]
        crop_img = img[y:y+h, x:x+w]
        p, gt = evaluator.prepare_for_detection(prediction_list, annot_list)
        iou = evaluator.iou_compute(p, gt)
        print(Path(img_path).name, iou)
        cv2.imwrite(str(Path(f'data/ears/detected/{D_TYPE}') / Path(img_path).name), crop_img)

def get_grouped_id_to_img_names(annot_path, ds_type):
    annots = np.loadtxt(annot_path, delimiter=',', dtype=object)
    tups = [(a[0].split('/')[1], int(a[1])) for a in annots if ds_type in a[0]]
    tups.sort(key=lambda x: x[1])
    grp = itertools.groupby(tups, lambda x: x[1])
    grp_dict = {}
    for k,v in grp:
        grp_dict[k] = [e[0] for e in list(v)]
    return grp_dict

def prepare_data_dir_structure(imgs_dir, new_dir, annot_path, ds_type):
    id_img_names = get_grouped_id_to_img_names(annot_path, ds_type)
    for id, img_names in id_img_names.items():
        id_dir_path = Path(new_dir) / str(id)
        id_dir_path.mkdir()

        for img in img_names:
            img_org = Path(imgs_dir) / img
            img_new = id_dir_path / img
            shutil.copy(str(img_org), str(img_new))

prepare_data_dir_structure(f'data/ears/detected/{D_TYPE}', f'data/ears/detected/{D_TYPE}_dir', 'data/perfectly_detected_ears/annotations/recognition/ids.csv', D_TYPE)