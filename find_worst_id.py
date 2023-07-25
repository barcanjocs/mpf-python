import numpy as np

def find_worst_id(diff_matrices, query, R_window):
    obs_vectors = []
    maxes = []
    for mt in diff_matrices :
        obs_vectors.append(mt[query])
    
    for v in obs_vectors :
        maxes.append(v.argmax())
    
    scores = np.zeros((len(diff_matrices)))

    for tech in range(len(diff_matrices) - 1) :
        for tech2 in range(tech, len(diff_matrices)) :
            d = abs(maxes[tech] - maxes[tech2])
            if d <= R_window :
                scores[tech] += 1
                scores[tech2] += 1
    
    if scores.sum() == 0 : # all are 0
        worst_id = 0
    
    elif scores.sum() == len(diff_matrices) - 1 : # all are max
        worst_id = 0
    
    else :
        best_id = scores.argmax()
        if best_id == 0 :
            worst_id = np.delete(scores, 0).argmin()
            
        elif best_id == 1 :
             worst_id = np.delete(scores, 1).argmin()
        
        elif best_id == 2 :
             worst_id = np.delete(scores, 2).argmin()
        
        else :
             worst_id = np.delete(scores, 3).argmin()
            
    return worst_id