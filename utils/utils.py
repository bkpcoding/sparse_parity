import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from typing import Dict, List
import torch.nn as nn
import random
import torch

def create_loss_animation(
    losses_subtasks: Dict[str, List[float]],
    log_steps: List[int],
    optimizer_name: str,
    verbose: bool = False
) -> str:
    """
    Create an animation of per-task loss over training steps.
    
    Parameters
    ----------
    losses_subtasks : Dict[str, List[float]]
        Dictionary mapping task IDs to their loss histories
    log_steps : List[int]
        List of steps at which losses were logged
    optimizer_name : str
        Name of the optimizer (for filename)
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    str
        Path to the saved GIF file
    """
    n_tasks = len(losses_subtasks)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Per-Task Loss Over Training Steps')
    ax.set_yscale('log')  # Use log scale for better visibility

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
    min_loss = max(1e-6, min(all_losses))  # Avoid zero for log scale
    max_loss = max(all_losses)
    ax.set_xlim(min_step, max_step)
    ax.set_ylim(min_loss * 0.9, max_loss * 1.1)

    # Animation update function
    def update(frame):
        for i in range(n_tasks):
            x_data = log_steps[:frame+1]
            y_data = losses_subtasks[str(i)][:frame+1]
            lines[i].set_data(x_data, y_data)
        return lines

    # Create animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(log_steps), 
        interval=50, 
        blit=True,
        repeat=False
    )

    # Save as GIF
    gif_path = f'per_task_loss_animation_{optimizer_name}.gif'
    ani.save(
        gif_path,
        writer='pillow', 
        fps=60,
        progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
    )
    plt.close(fig)
    
    if verbose:
        print(f"Animation saved to {gif_path}")
    
    return gif_path


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



def tasks_equal(tasks1, tasks2):
    """Compare two lists of tasks for equality."""
    if len(tasks1) != len(tasks2):
        return False
    return all(tuple(t1) == tuple(t2) for t1, t2 in zip(tasks1, tasks2))


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
