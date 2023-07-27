'''
Multi Process Fusion for visual place recognition.

This implementation does not require the actual VPR techniques but
only the difference/score matrices.
'''

import numpy as np
import pandas as pd
from mpf_run import mpf_run
from settings import AlgoSettings

algo_settings = AlgoSettings()

matrices_path = '../research/projects/A-MuSIC/data/score_vectors/'
techniques = ['HybridNet', 'CoHOG', 'HOG', 'CALC']
dataset = 'winter'
diff_matrices = []

# Load difference matrices
for t in techniques :
    filename = matrices_path + t + '_' + dataset + '.csv'
    diff_matrices.append(np.array(pd.read_csv(filename, header=None)))

mpf_run(diff_matrices, algo_settings)





