import cv2, sys, os

class Detector:

    cascades = [
        cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml')),
        cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))
    ]

    def __init__(self, scale_factor=1.05, min_neighbors=1):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, img):
        det_list = []
        for cascade in self.cascades:
            det_list.extend(cascade.detectMultiScale(img, self.scale_factor, self.min_neighbors))
        return det_list

# if __name__ == '__main__':
#     fname = sys.argv[1]
#     img = cv2.imread(fname)
#     detector = CascadeDetector()
#     detected_loc = detector.detect(img)
#     for x, y, w, h in detected_loc:
#         cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
#     cv2.imwrite(fname + '.detected.jpg', img)