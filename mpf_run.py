import numpy as np
import numpy.typing as npt
from viterbi_smart_dynamic_features import vit_sdf
from patch_normalize_hmm import patch_normalize_hmm
from mpf import AlgoSettings

def mpf_run(diff_matrices: list, algo_settings: AlgoSettings) :
    num_techniques = len(diff_matrices)
    num_queries = diff_matrices[0].size[0]
    num_places = diff_matrices[0].size[1]
    
    # Initialize transition matrix
    transition_matrix = np.zeros((num_queries, num_places))
    for j in range(num_queries) :
        for k in range(num_places) :
            if ((k-j) >= algo_settings.min_vel) and ((k-j) <= algo_settings.max_vel) :
                transition_matrix[j, k] = 1
            else :
                transition_matrix[j, k] = 0.001
    
    for query in range(num_queries) :
        for tech in range(num_techniques):
            # Normalise according to MPF paper/code
            diff_vector = np.array(diff_matrices[tech][query])
            mx = diff_vector.max()
            df = mx - diff_vector.min()
            
            for k in range(num_places):
                O_diff = ((mx - diff_vector[k])/df) - algo_settings.epsilon
            
                if O_diff < algo_settings.obs_th :
                    diff_matrices[tech][query][k] = algo_settings.epsilon
                else :
                    diff_matrices[tech][query][k] = O_diff


        if query > algo_settings.max_seq_len :
            S = np.arange(query - algo_settings.max_seq_len, query)
            
            seq, quality, newSeqLen = vit_sdf(S, transition_matrix, diff_matrices, algo_settings)

            