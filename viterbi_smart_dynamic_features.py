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
        obs = tech[S, :].T
        log_obs.append(np.log(obs))
    
    log_obs = np.array(log_obs)
    
    full_obs = np.zeros((num_places, tau))
    
    for q in range(tau) :
        worst_id = worst_ID_array[S[q]]
        for tech in range(len(diff_matrices)) :
            if tech != worst_id :
                full_obs[:, q] += log_obs[tech][:, q]
        
    min_values = full_obs.max(axis = 0)
    min_indices = full_obs.argmax(axis = 0)
    
    for q in range(tau) :
        window = np.arange(max(0, min_indices[q] - algo_settings.R_window),
                           min(num_places-1, min_indices[q] + algo_settings.R_window))
        not_window = np.setxor1d(np.arange(0, num_places-1), window)
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
    SS = np.zeros(tau)
    delta = np.int16(delta)
    H[:, 0] = 0
    
    if worst_ID_array[S[0]] == -1 :
        delta[:, 0] = log_obs[:, :, seq_start+1].sum(axis=0)
        
    else:
        delta[:, 0] = np.delete(log_obs, int(worst_ID_array[S[0]]), 0)[:, :, seq_start+1].sum(axis=0)
    
    for q in range(1, tau) :
        tmp = (delta[:, q - 1] + T)
        delta[:, q] = tmp.max(axis=0)
        H[:, q] = tmp.argmax(axis=0)
        
        if worst_ID_array[S[q]] == -1 :
            delta[:, q] = delta[:, q] + log_obs[:, :, q+seq_start].sum(axis=0)
            
        else :
            delta[:, q] = delta[:, q] + np.delete(log_obs, int(worst_ID_array[S[q]]), 0)[:, :, q+seq_start].sum(axis=0)
    SS[tau-1] = delta[:, tau-1].argmax()
    
    quality_total = 0
    
    for q in range(tau-2, -1, -1) :
        SS[q] = H[int(SS[q+1]), q+1]
        min_idx = SS[q+1]
        min_value = float(delta[int(SS[q+1]), q+1])
        
        window = np.arange(max(0, min_idx - algo_settings.R_window), 
                           min(min_idx+algo_settings.R_window, num_places-1))
       
        not_window = np.setxor1d(np.arange(0, num_places-1), window).astype(int)
        
        min_value_2nd = float(delta[not_window, q+1].max())
        
        quality = min_value / min_value_2nd
        quality_total += quality
    
    seq_len = tau
    
    return SS, quality_total, seq_len
    
    
    
    
    
    
        
    
    
    
    