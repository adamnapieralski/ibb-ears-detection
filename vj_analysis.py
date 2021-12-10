import numpy as np
import json, pickle
import itertools
import matplotlib.pyplot as plt

from detectors.cascade_detector.detector import Detector
from preprocessing.preprocess import Preprocess
from run_evaluation import EvaluateAll

class VjAnalysis:

    def __init__(self):
        self.scale_factors = []
        self.min_neighbors = []
        self.params = []
        self.info = {}
        self.eval = EvaluateAll()
        self.detector = Detector(scale_factor=1.015, min_neighbors=2)

    def run(self, **kwargs):
        def preprocess(img):
            pp = Preprocess()
            if not bool(kwargs):
                return img
            else:
                method = kwargs['method']
                if method == 'plain':
                    return img
                if method == 'blur':
                    if kwargs['type'] == 'median':
                        return pp.blur(img, kwargs['type'], kwargs['ksize'])
                    else:
                        return pp.blur(img, kwargs['type'], (kwargs['ksize'], kwargs['ksize']))
                if method == 'resize':
                    return pp.resize(img, dsize=(kwargs['dsize'], kwargs['dsize']))
                if method == 'contrast_brightness':
                    return pp.change_contrast_brightness(img, kwargs['alpha'], kwargs['beta'])

        res = self.eval.run_evaluation_vj_analysis(self.detector, preprocess)

        if kwargs['method'] not in self.info:
            self.info[kwargs['method']] = {}

        method = kwargs['method']

        if method == 'plain':
            self.info['plain'] = res
        if method == 'blur':
            self.info[method][kwargs['type'] + ',' + str(kwargs['ksize'])] = res
        if method == 'resize':
            self.info[method][str(kwargs['dsize'])] = res
        if method == 'contrast_brightness':
            self.info[method][str(kwargs['alpha']) + ',' + str(kwargs['beta'])] = res

        print(kwargs, res)

    def run_all(self, results_fname):
        for i, p in enumerate(self.params):
            self.run(**p)

            if i % 5 == 0:
                with open(results_fname, 'wt') as f:
                    json.dump(self.info, f, indent=2)
        with open(results_fname, 'wt') as f:
            json.dump(self.info, f, indent=2)

    def load_info(self, fname):
        with open(fname, 'rt') as f:
            self.info = json.load(f)

    def plot_blur(self, out_fname):
        vals = {'3': [], '5': [], '7': [], '9': [], '11': []}
        for k,v in self.info['blur'].items():
            vals[k.split(',')[1]].append(v)
        df = [v for v in vals.values()]

        figure, ax = plt.subplots(figsize=(7,7))
        ind = np.arange(3)
        width = 0.15
        rects = []
        for i in range(5):
            rect = ax.bar(ind + (i-2)*width, df[i], width = width)
            rects.append(rect)

        # rects1 = ax.bar(ind - )
        ax.set_xticks(ind, ('gaussian', 'box', 'median'))
        ax.legend(labels=['3', '5', '7', '9', '11'], title='Kernel size')

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 0.8*height,
                        '%.2f%%' % (height*100),
                        ha='center', va='bottom', rotation='vertical')
        for r in rects:
            autolabel(r)
        ax.set_title('VJ detection IoU score analysis - blur')
        figure.savefig(out_fname, facecolor='white')

    def plot_contrast_brightness(self, out_fname):
        vals = []
        cont = 0.25
        cont_vals = []
        for k,v in self.info['contrast_brightness'].items():
            if float(k.split(',')[0]) != cont:
                vals.append(cont_vals)
                cont_vals = []
                cont = float(k.split(',')[0])
            cont_vals.append(v)
        vals.append(cont_vals)
        vals = np.array(vals)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(vals)

        contrasts = [str(c) for c in list(np.arange(0.25, 2.25, 0.25))]
        brights = [str(b) for b in list(np.arange(-30, 40, 10))]

        ax.set_xticks(np.arange(len(brights)), labels=brights)
        ax.set_yticks(np.arange(len(contrasts)), labels=contrasts)

        for i in range(len(contrasts)):
            for j in range(len(brights)):
                text = ax.text(j, i, '{:.2f}%'.format(vals[i, j]*100),
                            ha="center", va="center", color="grey")

        ax.set_title('VJ detection IoU score analysis - contrast & brightness')
        ax.set_xlabel('Brightness (beta)')
        ax.set_ylabel('Contrast (alpha)')
        fig.savefig(out_fname, facecolor='white')

    def plot_params_search(self, params_search_res_data_path, out_fname, figsize=(15,10)):
        with open(params_search_res_data_path, 'rb') as f:
            info = pickle.load(f)

        sf = [round(v, 3) for v in info['scale_factors']]
        mn = [round(v, 2) for v in info['min_neihbors']]
        res = info['results']
        res = np.array(res).T

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(res)

        ax.set_yticks(np.arange(len(mn)), labels=mn)
        ax.set_xticks(np.arange(len(sf)), labels=sf)

        for i in range(len(mn)):
            for j in range(len(sf)):
                text = ax.text(j, i, '{:.2f}%'.format(res[i, j]*100),
                            ha="center", va="center", color="grey")

        ax.set_title('VJ detection IoU score analysis - parameters search (detailed)')
        ax.set_xlabel('Scale factor')
        ax.set_ylabel('Min neighbors')
        fig.savefig(out_fname, facecolor='white')

def define_initial_params(self):
    blur_params = [t for t in itertools.product(
        ['gaussian', 'box', 'median'], [3, 5, 7, 9, 11]
    )]

    cb_params = [t for t in itertools.product(
        list(np.arange(0.25, 2.25, 0.25)), list(np.arange(-30, 40, 10))
    )]

    self.params.append({ 'method': 'plain' })

    self.params.extend([
        { 'method': 'resize', 'dsize': p } for p in [360, 480]
    ])

    self.params.extend([
        { 'method': 'blur', 'type': p[0], 'ksize': p[1] } for p in blur_params
    ])
    self.params.extend([
        { 'method': 'contrast_brightness', 'alpha': p[0], 'beta': p[1] } for p in cb_params
    ])


if __name__ == '__main__':
    vja = VjAnalysis()

    vja.run_all('vj_analysis_2_cb.json')
    vja.load_info('results/vj_analysis.json')
    vja.plot_blur('results/fig/vj_analysis_blur.png')
    vja.plot_contrast_brightness('results/fig/vj_analysis_contrast_brightness.png')
    vja.plot_params_search('results/vj_params_search.pkl', 'results/fig/vj_analysis_params_search.png')
    vja.plot_params_search('results/vj_params_search_2.pkl', 'results/fig/vj_analysis_params_search_detailed.png', figsize=(30,5))
