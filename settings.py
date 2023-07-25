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