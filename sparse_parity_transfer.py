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

    D = 200000 # -1 for infinite data
    # D = -1
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

    n_ensembles = 1  # Add this new parameter for number of ensemble runs

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

def tasks_equal(tasks1, tasks2):
    """Compare two lists of tasks for equality."""
    if len(tasks1) != len(tasks2):
        return False
    return all(tuple(t1) == tuple(t2) for t1, t2 in zip(tasks1, tasks2))

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
                
                # Store per-task losses for the animation
                if 'losses_subtasks_one_bit' in ex.info and tasks_equal(Ss, get_one_bit_different_tasks(n, k, n_tasks)):
                    ex.info['log_steps_one_bit'].append(step)
                    for i in range(n_tasks):
                        x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i],
                                           sizes=[test_points_per_task], device=device, dtype=dtype)
                        y_i_pred = model(x_i)
                        ex.info['losses_subtasks_one_bit'][str(i)].append(loss_fn(y_i_pred, y_i).item())
                
                elif 'losses_subtasks_random' in ex.info and tasks_equal(Ss, get_random_tasks(n, k, n_tasks)):
                    ex.info['log_steps_random'].append(step)
                    for i in range(n_tasks):
                        x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i],
                                           sizes=[test_points_per_task], device=device, dtype=dtype)
                        y_i_pred = model(x_i)
                        ex.info['losses_subtasks_random'][str(i)].append(loss_fn(y_i_pred, y_i).item())
                
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
        lr, weight_decay, stop_early, device, dtype, log_freq, verbose, seed, 
        n_ensembles, _log):
    
    # Set random seeds
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
    
    def create_loss_animation(losses_subtasks, log_steps, method_name):
        """Create and save loss animation for a specific method."""
        if not log_steps:  # Check if log_steps is empty
            print(f"Warning: No loss data collected for {method_name} method")
            return None
        
        n_tasks = len(losses_subtasks)
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Per-Task Loss Over Training Steps ({method_name})')
        ax.set_yscale('log')
        
        # Generate colors for each task
        colors = cm.viridis(np.linspace(0, 1, n_tasks))
        
        # Initialize lines for each task
        lines = []
        for i in range(n_tasks):
            line, = ax.plot([], [], color=colors[i], alpha=0.5)
            lines.append(line)
        
        # Set axis limits
        all_losses = [loss for task_loss in losses_subtasks.values() for loss in task_loss]
        min_step = min(log_steps)
        max_step = max(log_steps)
        min_loss = max(1e-6, min(all_losses))
        max_loss = max(all_losses)
        ax.set_xlim(min_step, max_step)
        ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
        
        def update(frame):
            for i in range(n_tasks):
                x_data = log_steps[:frame+1]
                y_data = losses_subtasks[str(i)][:frame+1]
                lines[i].set_data(x_data, y_data)
            return lines
        
        ani = FuncAnimation(
            fig,
            update,
            frames=len(log_steps),
            interval=50,
            blit=True,
            repeat=False
        )
        
        # Save as GIF
        gif_path = f'per_task_loss_animation_{method_name}.gif'
        ani.save(
            gif_path,
            writer='pillow',
            fps=60,
            progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
        )
        ex.add_artifact(gif_path)
        plt.close(fig)
        return gif_path

    # Initialize lists to store results for each ensemble
    ensemble_results = defaultdict(list)
    
    for ensemble_idx in range(n_ensembles):
        print(f"\nRunning ensemble {ensemble_idx + 1}/{n_ensembles}")
        
        # Run one-bit difference comparison
        if comparison_mode in ['one_bit_diff', 'both']:
            Ss_one_bit = get_one_bit_different_tasks(n, k, n_tasks)
            print(f"One-bit difference tasks: {Ss_one_bit}")
            mlp_one_bit = create_model(n_tasks, n, width, depth, activation_fn, device, dtype)
            
            # Store losses for the first ensemble run only
            if ensemble_idx == 0:
                ex.info['losses_subtasks_one_bit'] = dict()
                ex.info['log_steps_one_bit'] = list()
                for i in range(n_tasks):
                    ex.info['losses_subtasks_one_bit'][str(i)] = list()
            
            steps_one_bit = train_model(
                mlp_one_bit, Ss_one_bit, n_tasks, n, k, alpha, offset, D,
                batch_size, lr, weight_decay, test_points, test_points_per_task,
                steps, device, dtype, log_freq, stop_early, verbose
            )
            ensemble_results['one_bit'].append(steps_one_bit)
            
            # Create animation for the first ensemble run only
            if ensemble_idx == 0:
                gif_path = create_loss_animation(
                    ex.info['losses_subtasks_one_bit'],
                    ex.info['log_steps_one_bit'],
                    'one_bit_diff'
                )
                if verbose:
                    print(f"One-bit difference animation saved to {gif_path}")
        
        # Run random tasks comparison
        if comparison_mode in ['random', 'both']:
            Ss_random = get_random_tasks(n, k, n_tasks)
            print(f"Random tasks: {Ss_random}")
            mlp_random = create_model(n_tasks, n, width, depth, activation_fn, device, dtype)
            
            # Store losses for the first ensemble run only
            if ensemble_idx == 0:
                ex.info['losses_subtasks_random'] = dict()
                ex.info['log_steps_random'] = list()
                for i in range(n_tasks):
                    ex.info['losses_subtasks_random'][str(i)] = list()
            
            steps_random = train_model(
                mlp_random, Ss_random, n_tasks, n, k, alpha, offset, D,
                batch_size, lr, weight_decay, test_points, test_points_per_task,
                steps, device, dtype, log_freq, stop_early, verbose
            )
            ensemble_results['random'].append(steps_random)
            
            # Create animation for the first ensemble run only
            if ensemble_idx == 0:
                gif_path = create_loss_animation(
                    ex.info['losses_subtasks_random'],
                    ex.info['log_steps_random'],
                    'random'
                )
                if verbose:
                    print(f"Random tasks animation saved to {gif_path}")
    
    # Calculate and print ensemble statistics
    print("\nEnsemble Statistics:")
    results = {}
    
    if comparison_mode in ['one_bit_diff', 'both']:
        one_bit_mean = np.mean(ensemble_results['one_bit'])
        one_bit_std = np.std(ensemble_results['one_bit'])
        print(f"One-bit difference tasks:")
        print(f"  Mean steps: {one_bit_mean:.2f}")
        print(f"  Std steps: {one_bit_std:.2f}")
        results['one_bit_mean'] = one_bit_mean
        results['one_bit_std'] = one_bit_std
        ex.info['convergence_steps_one_bit_mean'] = one_bit_mean
        ex.info['convergence_steps_one_bit_std'] = one_bit_std
    
    if comparison_mode in ['random', 'both']:
        random_mean = np.mean(ensemble_results['random'])
        random_std = np.std(ensemble_results['random'])
        print(f"Random tasks:")
        print(f"  Mean steps: {random_mean:.2f}")
        print(f"  Std steps: {random_std:.2f}")
        results['random_mean'] = random_mean
        results['random_std'] = random_std
        ex.info['convergence_steps_random_mean'] = random_mean
        ex.info['convergence_steps_random_std'] = random_std
    
    if comparison_mode == 'both':
        mean_diff = abs(one_bit_mean - random_mean)
        print(f"\nMean difference: {mean_diff:.2f} steps")
        results['mean_difference'] = mean_diff
        ex.info['convergence_mean_difference'] = mean_diff
        
        # Store raw ensemble results for further analysis if needed
        ex.info['ensemble_results'] = dict(ensemble_results)
    
    return results
