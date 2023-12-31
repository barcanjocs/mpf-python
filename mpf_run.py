import numpy as np
from viterbi_smart_dynamic_features import vit_sdf
from patch_normalize_hmm import patch_normalize_hmm
from settings import AlgoSettings
from find_worst_id import find_worst_id

def mpf_run(diff_matrices: list, algo_settings: AlgoSettings) :
    num_techniques = len(diff_matrices)
    num_queries = diff_matrices[0].shape[0]
    num_places = diff_matrices[0].shape[1]
    worst_id_array = np.zeros(num_queries)
    diff_matrices = np.array(diff_matrices)
    
    preds = []
    pred_scores = []
    quals = []
    
    normalized_matricies = np.zeros((len(diff_matrices), num_queries, num_places))
    
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
            
            diff_vector = np.array(diff_matrices[tech][query])
            mx = diff_vector.max()
            mn = diff_vector.min()
            df = mx - diff_vector.min()
            
            for k in range(num_places):
                # Normalisation is changed to comply with using vectors of similarities (bigger = better);
                # the original paper used vectors of differences (smaller = better)
                O_diff = ((diff_vector[k] - mn)/df) - algo_settings.epsilon 
            
                if O_diff < algo_settings.obs_th :
                    normalized_matricies[tech][query][k] = algo_settings.epsilon
                else :
                    normalized_matricies[tech][query][k] = O_diff

        worst_id_array[query] = find_worst_id(normalized_matricies, query, algo_settings.R_window)
        
        if query > algo_settings.max_seq_len :
            S = np.arange(query - algo_settings.max_seq_len, query)
            
            seq, quality, newSeqLength, scores = vit_sdf(S, transition_matrix, normalized_matricies, algo_settings, worst_id_array)

            quality = quality/newSeqLength
            
            id = seq[newSeqLength-1]

            preds.append(id)
            quals.append(quality)
            pred_scores.append(scores)
            
    return preds, quals, np.array(pred_scores)
        
    