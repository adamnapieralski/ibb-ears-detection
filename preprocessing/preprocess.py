import cv2
import numpy as np

class Preprocess:

    def __init__(self):
        self.mean_BGR = np.array([])

    def _get_mean_pixels(self, img):
        return np.mean(img, axis=(0,1))

    def fit_pixels_means(self, imgs):
        imgs_means = [self._get_mean_pixels(img) for img in imgs]
        self.mean_BGR = np.array(imgs_means).mean(axis=0)

    def subtract_pixels_means(self, imgs):
        for img in imgs:
            pass


    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize(self, img, dsize=(360, 360)):
        return cv2.resize(img, dsize)

    def blur(self, img, type: str, ksize: tuple = (5,5)):
        if type == 'box':
            return cv2.blur(img, ksize)
        elif type == 'gaussian':
            return cv2.GaussianBlur(img, ksize, 0)
        elif type == 'median':
            return cv2.medianBlur(img, ksize)

    def change_contrast_brightness(self, img, alpha, beta):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)

        # cv2.convertT

        # if value >= 0:
        #     lim = 255 - value
        #     v[v > lim] = 255
        #     v[v <= lim] += value
        # else:
        #     v[v < value] = 0
        #     v[v >= value] += value

        # final_hsv = cv2.merge((h, s, v))
        # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        # return img

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    # Automatic brightness and contrast optimization with optional histogram clipping
    def automatic_brightness_and_contrast(self, img, clip_hist_percent=25):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return (auto_result, alpha, beta)

    # img = cv2.imread('1.png')
    # auto_result, alpha, beta = automatic_brightness_and_contrast(img)
    # print('alpha', alpha)
    # print('beta', beta)
    # cv2.imshow('auto_result', auto_result)
    # cv2.imwrite('auto_result.png', auto_result)
    # cv2.imshow('img', img)
    # cv2.waitKey()