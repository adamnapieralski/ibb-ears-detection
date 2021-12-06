from pathlib import Path
import cv2


def convert_annotation_bbox_to_yolo(size, bbox):
    dh, dw = 1. / size[0], 1. / size[1]
    x = (bbox[0] + bbox[2] / 2.) * dw
    y = (bbox[1] + bbox[3] / 2.) * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return (x, y, w, h)

def get_annotations(annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

def save_yolo_format(annotations_dir, imgs_dir, out_dir):
    annotations_dir = Path(annotations_dir)
    annotations_files = [af for af in annotations_dir.iterdir()]
    annotations_files.sort()

    imgs_dir = Path(imgs_dir)
    imgs_files = [imf for imf in imgs_dir.iterdir()]
    imgs_files.sort()

    out_dir = Path(out_dir)

    for annot_file, img_file in zip(annotations_files, imgs_files):
        if annot_file.stem != img_file.stem:
            raise Exception('Annotation file not corresponding to image file')

        filename = annot_file.stem
        annot_list = get_annotations(annot_file)
        img = cv2.imread(str(img_file))

        str_data = ""
        for obj_bbox in annot_list:
            (x, y, w, h) = convert_annotation_bbox_to_yolo(img.shape, obj_bbox)
            str_data += '0 {} {} {} {}\n'.format(x, y, w, h)
        str_data = str_data[0:-1]

        with open(out_dir / (filename + '.txt'), 'w+') as f:
            f.write(str_data)