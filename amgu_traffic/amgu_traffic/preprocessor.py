import numpy as np

__all__ = ["LaneQeueueLength", "Vanila"]


def LaneQeueueLength(obs, waiting_cars_dim_idx):
    """Transform The State To New One,
        Wich Include Queue Length For Eahc Lane.

    Args:
        obs (np.ndarray): Env Observation.
        waiting_cars_dim_idx (int): Observation Dim Of Waiting Cars (Binary).

    Returns:
        np.ndarray: Transformed Observation
    """
    assert waiting_cars_dim_idx < len(obs.shape)

    waiting_cars_val = obs[waiting_cars_dim_idx]
    max_val = np.max(waiting_cars_val)
    min_val = np.min(waiting_cars_val)
    
    assert (min_val == 0 or min_val == 255) and (max_val == 0 or max_val == 255)

    return np.count_nonzero(waiting_cars_val, axis=-1)


def Vanila(obs):
    """Transform The State To The Same One.
    Args:
        obs (np.ndarray): Env Observation.

    Returns:
        np.ndarray: Same Observation
    """
    return obs
