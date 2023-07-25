'''
Multi Process Fusion for visual place recognition.

This implementation does not require the actual VPR techniques but
only the difference/score matrices.
'''

import numpy as np
import pandas as pd
from mpf_run import mpf_run
from dataclasses import dataclass

@dataclass
class AlgoSettings :
    max_seq_len = 20
    min_seq_len = 5
    obs_th = 0.5
    epsilon = 0.001
    min_vel = 0
    max_vel = 5
    qual_th = 0.1
    plot_th = 0.4
    R_window = 20
    normalise = True
    qROC_smooth = False

algo_settings = AlgoSettings()

matrices_path = ''
techniques = []
dataset = 'winter'
diff_matrices = []

# Load difference matrices
for t in techniques :
    filename = matrices_path + t + '_' + dataset + '.csv'
    diff_matrices.append(np.array(pd.read_csv(filename, header=None)))

mpf_run(diff_matrices, algo_settings)





