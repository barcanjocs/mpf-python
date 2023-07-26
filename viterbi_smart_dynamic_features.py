import numpy as np
from settings import AlgoSettings


def vit_sdf(S, T, diff_matrices, algo_settings: AlgoSettings, worst_ID_array) :
    tau = len(S) # Sequence length
    num_techs = len(diff_matrices)
    num_queries = diff_matrices[0].shape[0]
    num_places = diff_matrices[0].shape[1]
    quality = np.zeros(tau)
    qROC = np.zeros(tau)
    
    T = np.log(T)
    log_obs = []
    for tech in diff_matrices:
        obs = tech[S].T
        log_obs.append(np.log(obs))
    
    log_obs = np.array(log_obs)
    
    full_obs = np.zeros((num_places, tau))
    
    for q in range(tau) :
        worst_id = worst_ID_array[q]
        for tech in range(len(diff_matrices)) :
            if tech != worst_id :
                full_obs[:, q] += log_obs[tech][:, q]
        
    min_values = full_obs.max(axis = 1)
    min_indices = full_obs.argmax(axis = 1)
    
    for q in range(tau) :
        window = np.arange(max(1, min_indices[q] - algo_settings.R_window),
                           min(num_places, min_indices[q] + algo_settings.R_window))
        not_window = np.setxor1d(np.arange(0, num_places), window)
        min_value_2nd = full_obs[not_window, q].max()
        quality[q] = min_values[q] / min_value_2nd
        
        
    if algo_settings.qROC_smooth :
        pass
    else : 
        for q in range(1, tau-algo_settings.min_seq_len+1) :
            qROC[q - 1] = (quality[q] - quality[q-1])

    q_compare = qROC.min()
    seq_start = qROC.argmin()
    
    if abs(q_compare) < algo_settings.qual_th :
        seq_start = 0
        
    tau -= seq_start
    
    T = np.int16(T*1000)
    log_obs = np.int16(log_obs*1000)
    
    delta = np.zeros((num_places, tau))
    H = np.zeros((num_places, tau))
    SS = np.zeros((1, tau))
    delta = np.int16(delta)
    H[:, 1] = 0
    
    
    
    
    
        
    
    
    
    