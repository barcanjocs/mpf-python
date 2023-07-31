'''
Multi Process Fusion for visual place recognition.

This implementation does not require the actual VPR techniques but
only the difference/score matrices.
'''

import numpy as np
import pandas as pd
from mpf_run import mpf_run
from settings import AlgoSettings
import sklearn
import helpers

algo_settings = AlgoSettings()

matrices_path = '../research/projects/A-MuSIC/data/score_vectors/'
techniques = ['NetVLAD', 'CoHOG', 'HOG', 'CALC']
dataset = '17places_night'
numTemplates = 2000
diff_matrices = []

# Load difference matrices
for t in techniques :
    filename = matrices_path + t + '_' + dataset + '.csv'
    diff_matrices.append(np.array(pd.read_csv(filename, header=None)))

if dataset == "stlucia" :
    for i in range(len(diff_matrices)) :
        diff_matrices[i] = diff_matrices[i][50:, 50:]
    

if dataset == "17places_night" :
    for i in range(len(diff_matrices)) :
        diff_matrices[i] = diff_matrices[i][:2000, :2000]

preds, qualities, scores = mpf_run(diff_matrices, algo_settings)

selected_scores = np.array(scores).max(axis=1)

labels = np.arange(numTemplates - len(selected_scores), numTemplates)

p, r, auc = helpers.custom_label_pr_auc(preds, selected_scores, labels, 5)

helpers.pr_auc_to_file(p, r, [auc], "./data/MPF_" + dataset + ".csv")






