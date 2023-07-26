import numpy as np

def find_worst_id(diff_matrices, query, R_window):
    obs_vectors = []
    maxes = []
    for mt in diff_matrices :
        obs_vectors.append(mt[query])
        
    obs_vectors = np.array(obs_vectors)
    
    for v in obs_vectors :
        maxes.append(v.argmax())
    
    scores = np.zeros((len(diff_matrices)))

    for tech in range(len(diff_matrices) - 1) :
        for tech2 in range(tech + 1, len(diff_matrices)) :
            d = abs(maxes[tech] - maxes[tech2])
            if d <= R_window :
                scores[tech] += 1
                scores[tech2] += 1
    
    if scores.sum() == 0 : # all are 0
        return -1
        
    
    elif scores.sum() == (len(diff_matrices) - 1) * len(diff_matrices) : # all are max
        return -1
    
    else :
        best_id = scores.argmax()
        obs_vectors[best_id] = 1000000
        worst_id = obs_vectors[:, maxes[best_id]].argmin()
        
            
    return worst_id