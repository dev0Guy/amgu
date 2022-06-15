import enum
import math
import typing
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import collections
import numpy as np


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
