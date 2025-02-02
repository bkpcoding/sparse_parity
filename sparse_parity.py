import numpy as np
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm


from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("sparse-parity-v4-odm")
ex.captured_out_filter = apply_backspaces_and_linefeeds
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import math
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from samplers.sampler_factory import create_sampler


def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32):
    """Creates batch for testing/evaluation purposes.

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
        Inputs with shape (sum(sizes), n_tasks + n)
    y : torch.Tensor
        Labels with shape (sum(sizes),)
    """
    batch_x = torch.zeros((sum(sizes), n_tasks + n), dtype=dtype, device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    
    start_idx = 0
    for S, size, code in zip(Ss, sizes, codes):
        if size > 0:
            # Generate random bits for this task
            bits = torch.randint(0, 2, (size, n), dtype=dtype, device=device)
            
            # Calculate parity for the specific subset S
            y = torch.sum(bits[:, list(S)], dim=1) % 2
            
            # Create one-hot task encoding
            task_encoding = torch.zeros((size, n_tasks), dtype=dtype, device=device)
            task_encoding[:, code] = 1
            
            # Combine task encoding and bits
            x = torch.cat([task_encoding, bits], dim=1)
            
            # Add to batch
            batch_x[start_idx:start_idx + size] = x
            batch_y[start_idx:start_idx + size] = y
            
            start_idx += size
    
    return batch_x, batch_y

def get_one_bit_different_tasks(n, k, n_tasks):
    """Generate n_tasks where each consecutive task differs by exactly one position."""
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
    D = -1  # infinite data
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
    verbose = True
    sampler_type = 'odm'  # 'uniform' or 'odm'
    sampler_config = {
        'warmup_steps': 100,
        'ma_alpha': 0.9
    }

    # Add seed for reproducibility
    seed = 42

@ex.automain
def run(n_tasks, n, k, alpha, offset, D, width, depth, activation, test_points,
        test_points_per_task, steps, batch_size, lr, weight_decay, stop_early,
        device, dtype, log_freq, verbose, seed, sampler_type, sampler_config, _log):

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

    # Create model
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

    # Generate one-bit-different tasks
    Ss = get_one_bit_different_tasks(n, k, n_tasks)
    ex.info['Ss'] = Ss
    print(f"Generated tasks: {Ss}")

    # Initialize sampler based on type
    if sampler_type == 'uniform':
        probs = np.array([np.power(n, -alpha) for n in range(1+offset, n_tasks+offset+1)])
        probs = probs / np.sum(probs)
        cdf = np.cumsum(probs)
    else:  # odm
        sampler = create_sampler(
            sampler_type='odm',
            config={
                'n': n,
                'k': k,
                'n_tasks': n_tasks,
                'D': D,
                'alpha': alpha,
                'offset': offset,
                'batch_size': batch_size,
                'device': device,
                'warmup_steps': sampler_config['warmup_steps'],
                'ma_alpha': sampler_config['ma_alpha'],
                'tasks': Ss  # Pass the same tasks to ODM sampler
            }
        )

    # Initialize training components
    loss_fn = nn.CrossEntropyLoss()  # Remove reduction='none' here
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize metrics tracking
    ex.info['log_steps'] = []
    ex.info['accuracies'] = []
    ex.info['losses'] = []
    ex.info['losses_subtasks'] = {str(i): [] for i in range(n_tasks)}
    ex.info['accuracies_subtasks'] = {str(i): [] for i in range(n_tasks)}
    early_stop_triggers = []

    # Training loop
    for step in tqdm(range(steps)):
        if step % log_freq == 0:
            with torch.no_grad():
                # Test each task
                all_converged = True
                for i in range(n_tasks):
                    x_i, y_i = get_batch(n_tasks=n_tasks, n=n, 
                                       Ss=[Ss[i]], codes=[i],
                                       sizes=[test_points_per_task],
                                       device=device, dtype=dtype)
                    y_i_pred = mlp(x_i)
                    loss = loss_fn(y_i_pred, y_i).item()  # Now this will work
                    ex.info['losses_subtasks'][str(i)].append(loss)
                    if loss > 1e-6:
                        all_converged = False

                # Check for convergence
                if all_converged:
                    print(f"\nConverged at step {step} with all tasks below 1e-6 loss")
                    ex.info['convergence_step'] = step
                    break

        # Training step
        optimizer.zero_grad()
        if sampler_type == 'uniform':
            samples = np.searchsorted(cdf, np.random.rand(batch_size,))
            hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks-1))
            x, y_target = get_batch(n_tasks=n_tasks, n=n, Ss=Ss,
                                  codes=list(range(n_tasks)),
                                  sizes=hist, device=device, dtype=dtype)
        else:
            x, y_target = sampler.sample_batch()

        y_pred = mlp(x)
        
        if sampler_type == 'odm':
            # For ODM, we need per-sample losses
            loss_unreduced = nn.CrossEntropyLoss(reduction='none')(y_pred, y_target)
            # Compute per-task losses for ODM update
            task_losses = defaultdict(list)
            for loss_val, task_idx in zip(loss_unreduced, sampler.get_task_indices()):
                task_losses[task_idx].append(loss_val.item())
            avg_task_losses = {
                task_idx: np.mean(losses) 
                for task_idx, losses in task_losses.items()
            }
            sampler.update(avg_task_losses)
            loss = loss_unreduced.mean()
        else:
            loss = loss_fn(y_pred, y_target)
        
        loss.backward()
        optimizer.step()

    # Save results
    ex.info['final_losses'] = {
        i: ex.info['losses_subtasks'][str(i)][-1] 
        for i in range(n_tasks)
    }
    ex.info['sampler_type'] = sampler_type
    
    # Print final metrics
    if verbose:
        print(f"\nFinal losses per task:")
        for i in range(n_tasks):
            print(f"Task {i}: {ex.info['final_losses'][i]:.2e}")
        if 'convergence_step' in ex.info:
            print(f"Converged at step {ex.info['convergence_step']}")
        else:
            print("Did not converge within maximum steps")

    return ex.info['final_losses']
