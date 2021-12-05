import numpy as np
import pickle

from detectors.cascade_detector.detector import Detector
from run_evaluation import EvaluateAll

class ParamsSearch:

    def __init__(self):
        self.scale_factors = []
        self.min_neighbors = []
        self.results = np.array([])
        self.eval = EvaluateAll()

    def set_params(self, scale_factors, min_neighbors):
        self.scale_factors = scale_factors
        self.min_neighbors = min_neighbors
        self.results = np.zeros(shape=(len(scale_factors), len(min_neighbors)))

    def save_results(self, fname):
        full_results = {
            'scale_factors': self.scale_factors,
            'min_neihbors': self.min_neighbors,
            'results': self.results
        }
        with open(fname, 'wb') as f:
            pickle.dump(full_results, f)

    def run(self, save_results_on_each=False, results_fname=None):
        for j, mn in enumerate(self.min_neighbors):
            for i, sf in enumerate(self.scale_factors):
                detector = Detector(scale_factor=sf, min_neighbors=mn)
                res = self.eval.run_evaluation_vj_detector(detector)
                self.results[i, j] = res

                if save_results_on_each:
                    self.save_results(results_fname)
