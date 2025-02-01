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

def get_one_bit_different_tasks(n, k, n_tasks):
    """Generate n_tasks where each consecutive task differs by exactly one position.
    
    Parameters
    ----------
    n : int
        Length of bit string
    k : int
        Number of active bits in each task
    n_tasks : int
        Number of tasks to generate
    
    Returns
    -------
    list of tuples
        List of n_tasks tasks where each consecutive pair differs by one bit
    """
    tasks = []
    
    # Generate first task's bits
    current_S = sorted(random.sample(range(n), k))
    tasks.append(tuple(current_S))
    
    # Generate remaining tasks
    for _ in range(n_tasks - 1):
        next_S = current_S.copy()
        available_positions = list(set(range(n)) - set(current_S))
        
        # If no available positions, reshuffle completely
        if not available_positions:
            next_S = sorted(random.sample(range(n), k))
        else:
            bit_to_remove = random.choice(current_S)
            new_bit = random.choice(available_positions)
            next_S.remove(bit_to_remove)
            next_S.append(new_bit)
            next_S.sort()
        
        tasks.append(tuple(next_S))
        current_S = next_S
    
    return tasks

def get_random_tasks(n, k, n_tasks):
    """Generate n_tasks completely random tasks.
    
    Parameters
    ----------
    n : int
        Length of bit string
    k : int
        Number of active bits in each task
    n_tasks : int
        Number of tasks to generate
    
    Returns
    -------
    list of tuples
        List of n_tasks random tasks
    """
    tasks = []
    while len(tasks) < n_tasks:
        S = tuple(sorted(random.sample(range(n), k)))
        if S not in tasks:  # Ensure no duplicate tasks
            tasks.append(S)
    return tasks




@ex.config
def cfg():
    n_tasks = 5
    n = 50
    k = 4
    comparison_mode = 'both'  # 'one_bit_diff', 'random', or 'both'
    alpha = 1.5
    offset = 0

    # D = 200000 # -1 for infinite data
    D = -1
    width = 100
    depth = 2
    activation = 'ReLU'
    
    steps = 250000
    batch_size = 10000
    lr = 1e-3
    weight_decay = 0.0
    test_points = 30000
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

def create_model(n_tasks, n, width, depth, activation_fn, device, dtype):
    """Create and initialize the MLP model."""
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
    return nn.Sequential(*layers).to(device)

def train_model(model, Ss, n_tasks, n, k, alpha, offset, D, batch_size, lr, weight_decay,
                test_points, test_points_per_task, steps, device, dtype, log_freq, 
                stop_early=False, verbose=False):
    """Train the model and return number of steps to convergence."""
    
    probs = np.array([np.power(n, -alpha) for n in range(1+offset, n_tasks+offset+1)])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)
    
    test_batch_sizes = [int(prob * test_points) for prob in probs]
    
    if D != -1:
        samples = np.searchsorted(cdf, np.random.rand(D,))
        hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks-1))
        train_x, train_y = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), 
                                   sizes=hist, device='cpu', dtype=dtype)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(train_x, train_y, 
                                          batch_size=min(D, batch_size), shuffle=True)
        train_iter = cycle(train_loader)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    losses = []
    early_stop_triggers = []
    
    for step in tqdm(range(steps)):
        if step % log_freq == 0:
            with torch.no_grad():
                x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, 
                                   codes=list(range(n_tasks)), 
                                   sizes=test_batch_sizes, 
                                   device=device, dtype=dtype)
                y_i_pred = model(x_i)
                loss = loss_fn(y_i_pred, y_i).item()
                losses.append(loss)
                
                # Check for convergence
                if loss <= 1e-6:
                    if verbose:
                        print(f"Converged at step {step} with loss {loss}")
                    return step
                
                # Early stopping logic
                if stop_early:
                    if step > 4000 and len(losses) >= 2 and losses[-1] > losses[-2]:
                        early_stop_triggers.append(True)
                    else:
                        early_stop_triggers.append(False)
                    if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                        return step
                    early_stop_triggers = early_stop_triggers[-10:]
        
        optimizer.zero_grad()
        if D == -1:
            samples = np.searchsorted(cdf, np.random.rand(batch_size,))
            hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks-1))
            x, y_target = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, 
                                  codes=list(range(n_tasks)), 
                                  sizes=hist, device=device, dtype=dtype)
        else:
            x, y_target = next(train_iter)
            
        y_pred = model(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
    
    return steps  # Return max steps if convergence not reached

@ex.automain
def run(n_tasks, n, k, comparison_mode, alpha, offset, D, width, depth,
        activation, test_points, test_points_per_task, steps, batch_size,
        lr, weight_decay, stop_early, device, dtype, log_freq, verbose, seed, _log):
    
    # Set random seeds
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Set activation function
    if activation == 'ReLU':
        activation_fn = nn.ReLU
    elif activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"
    
    results = {}
    
    # Run one-bit difference comparison
    if comparison_mode in ['one_bit_diff', 'both']:
        Ss_one_bit = get_one_bit_different_tasks(n, k, n_tasks)
        print(f"One-bit difference tasks: {Ss_one_bit}")
        mlp_one_bit = create_model(n_tasks, n, width, depth, activation_fn, device, dtype)
        steps_one_bit = train_model(
            mlp_one_bit, Ss_one_bit, n_tasks, n, k, alpha, offset, D,
            batch_size, lr, weight_decay, test_points, test_points_per_task,
            steps, device, dtype, log_freq, stop_early, verbose
        )
        results['one_bit'] = steps_one_bit
    
    # Run random tasks comparison
    if comparison_mode in ['random', 'both']:
        Ss_random = get_random_tasks(n, k, n_tasks)
        print(f"Random tasks: {Ss_random}")
        mlp_random = create_model(n_tasks, n, width, depth, activation_fn, device, dtype)
        steps_random = train_model(
            mlp_random, Ss_random, n_tasks, n, k, alpha, offset, D,
            batch_size, lr, weight_decay, test_points, test_points_per_task,
            steps, device, dtype, log_freq, stop_early, verbose
        )
        results['random'] = steps_random
    
    # Print and store results
    if comparison_mode == 'both':
        print(f"\nConvergence Comparison:")
        print(f"One-bit difference tasks: {results['one_bit']} steps")
        print(f"Random tasks: {results['random']} steps")
        print(f"Difference: {abs(results['one_bit'] - results['random'])} steps")
        
        ex.info['convergence_steps_one_bit'] = results['one_bit']
        ex.info['convergence_steps_random'] = results['random']
        ex.info['convergence_difference'] = abs(results['one_bit'] - results['random'])
    elif comparison_mode == 'one_bit_diff':
        print(f"\nOne-bit difference tasks convergence: {results['one_bit']} steps")
        ex.info['convergence_steps_one_bit'] = results['one_bit']
    else:
        print(f"\nRandom tasks convergence: {results['random']} steps")
        ex.info['convergence_steps_random'] = results['random']
    
    return results
