import numpy as np

__all__ = ['lane_length','vanila']


def lane_length(obs,waiting_cars_dim_idx):

    assert waiting_cars_dim_idx < len(obs.shape)
    
    waiting_cars_val = obs[waiting_cars_dim_idx]
    max_val = np.max(waiting_cars_val)
    min_val = np.min(waiting_cars_val)
    
    assert min_val == 0 and (max_val == 0 or max_val == 1)
    
    return np.count_nonzero(waiting_cars_val,axis=-1)

def vanila(obs): 
    return obs
