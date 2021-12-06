import numpy as np
import pickle
from timeit import default_timer as timer

from detectors.cascade_detector.detector import Detector
from run_evaluation import EvaluateAll

class ParamsSearch:

    def __init__(self):
        self.scale_factors = []
        self.min_neighbors = []
        self.results = np.array([])
        self.timers = np.array([])
        self.eval = EvaluateAll()

    def set_params(self, scale_factors, min_neighbors):
        self.scale_factors = scale_factors
        self.min_neighbors = min_neighbors
        self.results = np.zeros(shape=(len(scale_factors), len(min_neighbors)))

    def save_results(self, fname):
        full_results = {
            'scale_factors': self.scale_factors,
            'min_neihbors': self.min_neighbors,
            'results': self.results,
            'timers': self.timers
        }
        with open(fname, 'wb') as f:
            pickle.dump(full_results, f)

    def run(self, save_results_on_each=False, results_fname=None):
        for j, mn in enumerate(self.min_neighbors):
            for i, sf in enumerate(self.scale_factors):
                detector = Detector(scale_factor=sf, min_neighbors=mn)
                start = timer()
                res = self.eval.run_evaluation_vj_detector(detector)
                elapsed = timer() - start

                self.results[i, j] = res
                self.timers = elapsed

                print('SF: {:.2f}\tMN: {}\tres: {:.4f}\ttime: {:.2f}'.format(sf, mn, res, elapsed))

                if save_results_on_each and i % 10 == 0:
                    self.save_results(results_fname)


if __name__ == '__main__':
    ps = ParamsSearch()
    ps.set_params(
        scale_factors=list(np.arange(1.05, 2.05, 0.05)),
        min_neighbors=[0, 1, 2, 3, 4, 5]
    )

    ps.run(
        save_results_on_each=True,
        results_fname='results/vj_params_search.pkl'
    )