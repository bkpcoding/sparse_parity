#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once.

Comments
    - now does sampling for everything except the test batch -- frequencies of subtasks are exactly distributed within test batch
    - now allows for early stopping
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
# import scipy.stats
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("sparse-parity-v4")
ex.captured_out_filter = apply_backspaces_and_linefeeds

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32):
    """Creates batch. 

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of int
        The subtask indices which the batch will consist of
    sizes : list of int
        Number of samples for each subtask
    device : str
        Device to put batch on.
    dtype : torch.dtype
        Data type to use for input x. Output y is torch.int64.

    Returns
    -------
    x : torch.Tensor
        inputs
    y : torch.Tensor
        labels
    """
    batch_x = torch.zeros((sum(sizes), n_tasks+n), dtype=dtype, device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    start_i = 0
    for (S, size, code) in zip(Ss, sizes, codes):
        if size > 0:
            x = torch.randint(low=0, high=2, size=(size, n), dtype=dtype, device=device)
            y = torch.sum(x[:, S], dim=1) % 2
            x_task_code = torch.zeros((size, n_tasks), dtype=dtype, device=device)
            x_task_code[:, code] = 1
            x = torch.cat([x_task_code, x], dim=1)
            batch_x[start_i:start_i+size, :] = x
            batch_y[start_i:start_i+size] = y
            start_i += size
    return batch_x, batch_y
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    n_tasks = 10
    n = 50
    k = 3
    alpha = 1.5
    offset = 0

    D = 500000 # -1 for infinite data

    width = 100
    depth = 2
    activation = 'ReLU'
    
    steps = 250000
    batch_size = 10000
    lr = 1e-3
    weight_decay = 0.0
    test_points = 3000
    test_points_per_task = 1000
    stop_early = False
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    log_freq = max(1, steps // 1000)
    verbose=False

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(n_tasks,
        n,
        k,
        alpha,
        offset,
        D,
        width,
        depth,
        activation,
        test_points,
        test_points_per_task,
        steps,
        batch_size,
        lr,
        weight_decay,
        stop_early,
        device,
        dtype,
        log_freq,
        verbose,
        seed,
        _log):

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if activation == 'ReLU':
        activation_fn = nn.ReLU
    elif activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"

    # create model
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(n_tasks + n, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 2))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)
    _log.debug("Created model.")
    _log.debug(f"Model has {sum(t.numel() for t in mlp.parameters())} parameters") 
    ex.info['P'] = sum(t.numel() for t in mlp.parameters())

    Ss = []
    for _ in range(n_tasks * 10):
        S = tuple(sorted(list(random.sample(range(n), k))))
        if S not in Ss:
            Ss.append(S)
        if len(Ss) == n_tasks:
            break
    assert len(Ss) == n_tasks, "Couldn't find enough subsets for tasks for the given n, k"
    ex.info['Ss'] = Ss

    probs = np.array([np.power(n, -alpha) for n in range(1+offset, n_tasks+offset+1)])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    test_batch_sizes = [int(prob * test_points) for prob in probs]
    # _log.debug(f"Total batch size = {sum(batch_sizes)}")

    if D != -1:
        samples = np.searchsorted(cdf, np.random.rand(D,))
        hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks))
        train_x, train_y = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=hist, device='cpu', dtype=dtype)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(train_x, train_y, batch_size=min(D, batch_size), shuffle=True)
        train_iter = cycle(train_loader)
        ex.info['D'] = D
    else:
        ex.info['D'] = steps * batch_size

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    ex.info['log_steps'] = list()
    ex.info['accuracies'] = list()
    ex.info['losses'] = list()
    ex.info['losses_subtasks'] = dict()
    ex.info['accuracies_subtasks'] = dict()
    for i in range(n_tasks):
        ex.info['losses_subtasks'][str(i)] = list()
        ex.info['accuracies_subtasks'][str(i)] = list()
    early_stop_triggers = []
    histories = []
    current_hist = np.zeros(n_tasks)
    step_interval = 10000
    histories.append(current_hist.copy())  # Initial zero distribution

    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=test_batch_sizes, device=device, dtype=dtype)
                y_i_pred = mlp(x_i)
                labels_i_pred = torch.argmax(y_i_pred, dim=1)
                ex.info['accuracies'].append(torch.sum(labels_i_pred == y_i).item() / test_points) 
                ex.info['losses'].append(loss_fn(y_i_pred, y_i).item())
                for i in range(n_tasks):
                    x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i], sizes=[test_points_per_task], device=device, dtype=dtype)
                    y_i_pred = mlp(x_i)
                    ex.info['losses_subtasks'][str(i)].append(loss_fn(y_i_pred, y_i).item())
                    labels_i_pred = torch.argmax(y_i_pred, dim=1)
                    ex.info['accuracies_subtasks'][str(i)].append(torch.sum(labels_i_pred == y_i).item() / test_points_per_task)
                ex.info['log_steps'].append(step)
            if stop_early:
                if step > 4000 and len(ex.info['losses']) >= 2 \
                    and ex.info['losses'][-1] > ex.info['losses'][-2]:
                    early_stop_triggers.append(True)
                else:
                    early_stop_triggers.append(False)
                if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                    break
                early_stop_triggers = early_stop_triggers[-10:]
        optimizer.zero_grad()
        if D == -1:
            samples = np.searchsorted(cdf, np.random.rand(batch_size,))
            hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks))
            current_hist += hist  # Accumulate the histogram
            
            if step > 0 and step % step_interval == 0:
                # Store and reset histogram every step_interval
                histories.append(current_hist.copy())
                current_hist = np.zeros(n_tasks)
            
            x, y_target = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=hist, device=device, dtype=dtype)
        else:
            x, y_target = next(train_iter)
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()

    log_steps = ex.info['log_steps']
    losses_subtasks = ex.info['losses_subtasks']
    n_tasks = len(losses_subtasks)

    # Create a figure and axis
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    ax_loss.set_xlabel('Training Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Per-Task Loss Over Training Steps')
    ax_loss.set_yscale('log')  # Use log scale for better visibility

    # Generate colors for each task
    colors = cm.viridis(np.linspace(0, 1, n_tasks))

    # Initialize lines for each task
    lines = []
    for i in range(n_tasks):
        line, = ax_loss.plot([], [], color=colors[i], alpha=0.5)
        lines.append(line)

    # Set axis limits
    all_losses = [loss for task_loss in losses_subtasks.values() for loss in task_loss]
    min_step = min(log_steps)
    max_step = max(log_steps)
    min_loss = max(1e-6, min(all_losses))  # Avoid zero for log scale
    max_loss = max(all_losses)
    ax_loss.set_xlim(min_step, max_step)
    ax_loss.set_ylim(min_loss * 0.9, max_loss * 1.1)

    # Animation update function
    def update(frame):
        for i in range(n_tasks):
            x_data = log_steps[:frame+1]
            y_data = losses_subtasks[str(i)][:frame+1]
            lines[i].set_data(x_data, y_data)
        return lines

    # Create animation
    ani = FuncAnimation(
        fig_loss, 
        update, 
        frames=len(log_steps), 
        interval=50, 
        blit=True,
        repeat=False
    )

    # Save as GIF
    gif_path = 'per_task_loss_animation_fixed_D.gif'
    ani.save(
        gif_path,
        writer='pillow', 
        fps=60,
        progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
    )
    ex.add_artifact(gif_path)
    plt.close(fig_loss)

    # Histogram Animation
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.set_xlabel('Task ID')
    ax_hist.set_ylabel('Number of Samples')
    ax_hist.set_title('Sample Distribution per Task')
    
    # Add theoretical distribution line
    task_ids = np.arange(n_tasks)
    samples_per_interval = (batch_size * step_interval)
    theoretical_dist = probs * samples_per_interval  # Scale probabilities to match accumulated samples
    ax_hist.plot(task_ids, theoretical_dist, 'r--', label='Theoretical Distribution')
    
    n_frames = len(histories)
    bar_width = 0.8
    bars = ax_hist.bar(range(n_tasks), histories[0], width=bar_width, color=colors, alpha=0.7)
    ax_hist.set_xticks(range(n_tasks))
    ax_hist.set_xlim(-0.5, n_tasks - 0.5)
    max_height = max(max(max(h) for h in histories), max(theoretical_dist))
    ax_hist.set_ylim(0, max_height * 1.1)
    ax_hist.legend()

    def update_hist(frame):
        for bar, height in zip(bars, histories[frame]):
            bar.set_height(height)
        steps_shown = frame * step_interval
        ax_hist.set_title(f'Sample Distribution per Task (Steps: {steps_shown:,})')
        return bars

    ani_hist = FuncAnimation(
        fig_hist,
        update_hist,
        frames=n_frames,
        interval=50,
        blit=False,
        repeat=False
    )

    hist_gif_path = 'sample_distribution_animation_fixed_D.gif'
    ani_hist.save(
        hist_gif_path,
        writer='pillow',
        fps=60,
        progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
    )
    ex.add_artifact(hist_gif_path)
    plt.close(fig_hist)

    # Print final metrics
    if verbose:
        print(f"\nFinal average accuracy: {ex.info['accuracies'][-1]:.4f}")
        print(f"Animation saved to {gif_path}")
        print(f"Histogram animation saved to {hist_gif_path}")
