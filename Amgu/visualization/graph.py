import enum
import math
import typing
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.stats as stats



def line_graph(
    evaluation_gen: typing.Generator,
    gen_size: int,
    yield_size: int,
    file_path: str,
    window_size=100,
    interval=10,
):
    """
    Args:
        evaluation_gen (typing.Generator): _description_
        gen_size (int): _description_
        yield_size (int): _description_
        file_path (str): _description_
        window_size (int): _description_. Defaults to 100.
        interval (int): _description_. Defaults to 10.
    """
    n_col = math.floor(math.sqrt(yield_size))
    n_row = math.ceil(yield_size / n_col)
    fig, axes = plt.subplots(n_col, n_row, figsize=(12, 6), facecolor=("#1e2121"))
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    iterator = iter(evaluation_gen)
    ep_rewards = np.zeros((yield_size, gen_size))
    avg_time_array = np.zeros(yield_size)
    live_reward = [collections.deque(np.zeros(window_size)) for _ in range(yield_size)]
    iter_n = [0]

    def online_visibility(b):
        iter_val = next(iterator, None)
        if iter_val is None:
            return
        for model_idx, val in enumerate(iter_val):
            ep_rewards[model_idx, iter_n[0]] = val.reward
            avg_time_array[model_idx] = val.waiting_time
            live_reward[model_idx].append(val.reward)
            # remove last window data
            live_reward[model_idx].popleft()
        for idx, ax in enumerate(axes.flatten()):
            # clear all axes for new draw
            ax.cla()
            # ax design
            ax.grid(color="#505454", linestyle="solid")
            ax.plot(live_reward[idx], color="#82d16b")
            ax.set_title("Reward")
            # plot scatter on the end of graph
            dot_pos = (
                len(live_reward[idx]),
                live_reward[idx][-1],
            )
            ax.scatter(*dot_pos, color="#82d16b")
            ax.text(*dot_pos, "{}".format(live_reward[idx][-1]), color="w")
            ax.set_ylim(-300, 0)
        iter_n[0] += 1

    ani = FuncAnimation(fig, online_visibility, interval=interval)
    plt.show()
    ani.save(file_path)

def offline_state_visualization():
    pass

def offline_attack_analysis(
    evaluation_gen: typing.Generator,
    gen_size: int,
    yield_size: int,
    file_path: str,
    window_size=100,
    interval=10,
):
    n_col = math.floor(math.sqrt(yield_size))
    n_row = math.ceil(yield_size / n_col)
    fig, axes = plt.subplots(n_col, n_row, figsize=(16, 5), facecolor=("#1e2121"),sharex = True)
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    iterator = iter(evaluation_gen)
    iter_n = [0]
    # create for each model
    division_info = [[] for i in range(yield_size)]
    division_attack_info = [[] for i in range(yield_size)]
    saver = [0]
    def state_attack_differ(b):
        iter_val = next(iterator, None)
        if iter_val is None:
            return
        tmp = axes.flatten()
        for model_idx, meta in enumerate(iter_val):
            ax = tmp[model_idx]
            obs = meta.org_obs
            attacked_obs = meta.attack_obs
            state_division = meta.state_division
            saver[0] = state_division.keys()
            prev = 0
            org_lst, attacked_lst = [] , []
            for idx,(key, val) in enumerate(state_division.items()):
                if idx >= len(division_info[model_idx]):
                    division_attack_info[model_idx].append([])
                    division_info[model_idx].append([])
                if key == 'Multiplyer':
                    multiplier = val
                    continue
                size, norm = val
                inner_val = (obs[prev:prev+size]) / multiplier * norm
                inner_attack_val = (attacked_obs[prev:prev+size]) / multiplier * norm
                # delta = np.abs(inner_val - inner_attack_val)
                division_info[model_idx][idx].append(inner_val)
                division_attack_info[model_idx][idx].append(inner_attack_val)
                org_lst.append(inner_val)
                attacked_lst.append(inner_attack_val)
                prev += size  
            y1 = np.concatenate(org_lst, axis=0)
            y2 = np.concatenate(attacked_lst, axis=0)
            x1 = np.array(list(range(len(y1))))
            ax.cla()
            ax.grid(color="#505454", linestyle="solid")
            ax.plot(x1,y1,label="Original",alpha=1) 
            ax.plot(x1,y2,label="Attack",alpha=1) 
            
            ax.fill_between(x1,0,y1,alpha=.2)
            ax.fill_between(x1,0,y2,alpha=.2)
            
            ax.legend()
        iter_n[0] += 1
    ani = FuncAnimation(fig, state_attack_differ, interval=interval)
    plt.show()
    ani.save(file_path)
    saver = list(saver[0])
    all_info = [[],[]]
    num_dvision = 0
    for model_idx in range(len(division_info)):
        org_model_info = division_info[model_idx]
        attack_model_info = division_attack_info[model_idx]
        org_division_data = {}
        attack_division_data = {}
        num_dvision = len(org_model_info)
        for key_idx in range(len(org_model_info)):
            org_division = org_model_info[key_idx]
            attack_division = attack_model_info[key_idx]
            if len(attack_division) == 0:
                continue
            org_division_data[saver[key_idx]] = {
                "mean": np.mean(org_division),
                "std": np.std(org_division),
                "min": np.min(org_division),
                "max": np.max(org_division),
            }
            attack_division_data[saver[key_idx]] = {
                "mean": np.mean(attack_division),
                "std": np.std(attack_division),
                "min": np.min(attack_division),
                "max": np.max(attack_division),
            }
        all_info[0].append(org_division_data)
        all_info[1].append(attack_division_data)
    num_dvision = (num_dvision-1) * 2
    n_col = math.floor(math.sqrt(num_dvision))
    n_row = math.ceil(num_dvision / n_col)
    fig, axes = plt.subplots(n_col, n_row, figsize=(16, 5), facecolor=("#1e2121"),sharex = True)
    fig.tight_layout() 
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    axes = axes.flatten()
    def normlize_plot(ax,std,mu):
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        y = stats.norm.pdf(x, mu, std)
        ax.plot(x, y)
        ax.fill_between(x,0,y,alpha=.3)
    org_info = all_info[0][0]
    attack_info = all_info[1][0]
    saver = list(org_info.keys())
    for idx,ax in enumerate(axes):
        key = saver[idx//2]
        if idx % 2: 
            bar_width = 1
            ax.bar([0],attack_info[key]["max"],width=bar_width)
            continue
        idx = idx//2
        ax.set_title(key)
        ax.grid(color="#505454", linestyle="solid")
        if org_info[key]["std"] == 0 :
            continue
        normlize_plot(ax,org_info[key]["std"],org_info[key]["mean"])        
        if attack_info[key]["std"] == 0 :
            continue
        normlize_plot(ax,attack_info[key]["std"],attack_info[key]["mean"])
    plt.show()