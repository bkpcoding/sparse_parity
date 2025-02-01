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

class EfficientODMSampler:
    """
    Efficient implementation of Online Data Mixing for sparse parity tasks.
    Pre-generates data per task and uses Exp3 for adaptive sampling.
    """
    def __init__(
        self,
        n: int,               # Total number of bits
        k: int,               # Size of each subset
        n_tasks: int,         # Number of different tasks
        D: int,               # Total dataset size (-1 for infinite)
        alpha: float = 1.5,   # Power law exponent
        offset: int = 0,      # Power law offset
        batch_size: int = 64,
        device: str = 'cpu',
        warmup_steps: int = 100,
        ma_alpha: float = 0.9  # Moving average parameter for rewards
    ):
        self.n = n
        self.k = k
        self.n_tasks = n_tasks
        self.D = D
        self.batch_size = batch_size
        self.device = device
        self.warmup_steps = warmup_steps
        self.ma_alpha = ma_alpha
        self.step = 0
        
        # Initialize tasks with overlapping bits for transfer learning
        self.tasks = []
        base_bits = sorted(list(np.random.choice(range(n), k, replace=False)))
        
        # First task uses k-1 bits
        self.tasks.append(tuple(base_bits[:k-1]))
        
        # Each subsequent task adds one new bit while keeping most previous bits
        available_bits = set(range(n)) - set(base_bits)
        for i in range(1, n_tasks):
            if i < len(base_bits):
                # Use the next bit from base_bits
                new_bits = base_bits[:k-1] + [base_bits[i]]
            else:
                # Add a new random bit
                new_bit = np.random.choice(list(available_bits))
                available_bits.remove(new_bit)
                new_bits = base_bits[i-len(base_bits):k-1] + [new_bit]
            
            self.tasks.append(tuple(sorted(new_bits)))

        # Calculate initial power law probabilities
        probs = np.array([np.power(n, -alpha) for n in range(1+offset, n_tasks+offset+1)])
        self.probs = probs / np.sum(probs)
        
        # Initialize Exp3 parameters
        self.estimated_rewards = np.zeros(n_tasks)
        
        # Pre-generate data per task
        if D != -1:
            self.finite_data = True
            self._generate_task_datasets()
        else:
            self.finite_data = False
            
        # Track losses per batch
        self.batch_losses = defaultdict(list)
        self.current_task_indices = []
    
    def _generate_task_datasets(self):
        """Generate fixed datasets for each task."""
        self.task_data = {}
        samples_per_task = np.random.multinomial(self.D, self.probs)
        
        for task_idx, n_samples in enumerate(samples_per_task):
            if n_samples > 0:
                # Generate random bits for this task
                bits = torch.randint(0, 2, (n_samples, self.n), device=self.device)
                # Add task encoding
                task_code = torch.zeros((n_samples, self.n_tasks), device=self.device)
                task_code[:, task_idx] = 1
                # Combine and store with targets
                x = torch.cat([task_code, bits], dim=1)
                y = torch.sum(bits[:, list(self.tasks[task_idx])], dim=1) % 2
                self.task_data[task_idx] = {
                    'x': x, 'y': y,
                    'position': 0,  # Track position in dataset
                    'size': n_samples
                }
    
    def _compute_exploration_rate(self) -> float:
        """Compute exploration rate for current step."""
        if self.step < self.warmup_steps:
            return 0.0
        t = max(1, self.step - self.warmup_steps)
        return min(1/self.n_tasks, math.sqrt(math.log(self.n_tasks)/(self.n_tasks * t)))
    
    def get_sampling_distribution(self) -> np.ndarray:
        """Calculate current sampling distribution using Exp3."""
        if self.step < self.warmup_steps:
            return self.probs
            
        exploration_rate = self._compute_exploration_rate()
        exp_rewards = np.exp(exploration_rate * self.estimated_rewards)
        gibbs_weights = exp_rewards / np.sum(exp_rewards)
        distribution = (1 - self.n_tasks * exploration_rate) * gibbs_weights + exploration_rate
        
        return distribution
    
    def _get_task_batch(self, task_idx: int, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of samples for a specific task."""
        if self.finite_data:
            task_data = self.task_data[task_idx]
            pos = task_data['position']
            n_samples = task_data['size']
            
            # Reset position if needed
            if pos + size > n_samples:
                task_data['position'] = 0
                pos = 0
                
            # Get batch
            x = task_data['x'][pos:pos+size]
            y = task_data['y'][pos:pos+size]
            task_data['position'] = (pos + size) % n_samples
            
            return x, y
        else:
            # Generate new data for infinite case
            bits = torch.randint(0, 2, (size, self.n), device=self.device)
            task_code = torch.zeros((size, self.n_tasks), device=self.device)
            task_code[:, task_idx] = 1
            x = torch.cat([task_code, bits], dim=1)
            y = torch.sum(bits[:, list(self.tasks[task_idx])], dim=1) % 2
            return x, y
    
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of data using current distribution."""
        # Get current sampling distribution
        distribution = self.get_sampling_distribution()
        
        # Sample tasks according to distribution
        task_counts = np.random.multinomial(self.batch_size, distribution)
        
        # Collect batches from each sampled task
        x_batches = []
        y_batches = []
        self.current_task_indices = []
        
        for task_idx, count in enumerate(task_counts):
            if count > 0:
                x, y = self._get_task_batch(task_idx, count)
                x_batches.append(x)
                y_batches.append(y)
                self.current_task_indices.extend([task_idx] * count)
        
        # Combine batches
        x = torch.cat(x_batches, dim=0)
        y = torch.cat(y_batches, dim=0)
        
        # Shuffle
        perm = torch.randperm(x.size(0))
        return x[perm], y[perm]
    
    def update(self, losses: Dict[int, float]):
        """Update estimated rewards based on observed losses."""
        if self.step < self.warmup_steps:
            self.step += 1
            return
            
        distribution = self.get_sampling_distribution()
        
        # Update estimated rewards using importance weighted rewards
        for task_idx, loss in losses.items():
            weighted_reward = loss / distribution[task_idx]
            self.estimated_rewards[task_idx] = (
                self.ma_alpha * self.estimated_rewards[task_idx] + 
                (1 - self.ma_alpha) * weighted_reward
            )
        
        self.step += 1

    def get_task_indices(self) -> List[int]:
        """Get indices of tasks in the current batch."""
        return self.current_task_indices



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

    D = 200000  # -1 for infinite data

    width = 100
    depth = 2
    activation = 'ReLU'
    
    steps = 1000
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
    
    # Add seed for reproducibility
    seed = 42


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

    # Initialize sampler
    sampler = EfficientODMSampler(
        n=n,
        k=k,
        n_tasks=n_tasks,
        D=D,
        alpha=alpha,
        offset=offset,
        batch_size=batch_size,
        device=device
    )
    ex.info['Ss'] = sampler.tasks

    # Initialize training components
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Use none reduction to get per-sample losses
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize metrics tracking
    ex.info['log_steps'] = []
    ex.info['accuracies'] = []
    ex.info['losses'] = []
    ex.info['losses_subtasks'] = {str(i): [] for i in range(n_tasks)}
    ex.info['accuracies_subtasks'] = {str(i): [] for i in range(n_tasks)}
    early_stop_triggers = []

    # Training loop
    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                # Test overall performance
                test_batch_sizes = [test_points // n_tasks] * n_tasks
                x_test, y_test = get_batch(
                    n_tasks=n_tasks, n=n, 
                    Ss=sampler.tasks, 
                    codes=list(range(n_tasks)), 
                    sizes=test_batch_sizes, 
                    device=device, 
                    dtype=dtype
                )
                y_pred = mlp(x_test)
                labels_pred = torch.argmax(y_pred, dim=1)
                ex.info['accuracies'].append(torch.sum(labels_pred == y_test).item() / test_points)
                ex.info['losses'].append(loss_fn(y_pred, y_test).mean().item())

                # Test per-task performance
                for i in range(n_tasks):
                    x_i, y_i = get_batch(
                        n_tasks=n_tasks, n=n,
                        Ss=[sampler.tasks[i]], 
                        codes=[i],
                        sizes=[test_points_per_task],
                        device=device,
                        dtype=dtype
                    )
                    y_i_pred = mlp(x_i)
                    ex.info['losses_subtasks'][str(i)].append(loss_fn(y_i_pred, y_i).mean().item())
                    labels_i_pred = torch.argmax(y_i_pred, dim=1)
                    ex.info['accuracies_subtasks'][str(i)].append(
                        torch.sum(labels_i_pred == y_i).item() / test_points_per_task
                    )
                ex.info['log_steps'].append(step)

            # Early stopping check
            if stop_early:
                if step > 4000 and len(ex.info['losses']) >= 2 \
                    and ex.info['losses'][-1] > ex.info['losses'][-2]:
                    early_stop_triggers.append(True)
                else:
                    early_stop_triggers.append(False)
                if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                    break
                early_stop_triggers = early_stop_triggers[-10:]

        # Training step
        optimizer.zero_grad()
        x, y_target = sampler.sample_batch()
        y_pred = mlp(x)
        losses = loss_fn(y_pred, y_target)
        
        # Compute per-task losses for ODM update
        task_losses = defaultdict(list)
        for loss, task_idx in zip(losses, sampler.get_task_indices()):
            task_losses[task_idx].append(loss.item())
        
        # Average losses per task
        avg_task_losses = {
            task_idx: np.mean(losses) 
            for task_idx, losses in task_losses.items()
        }
        
        # Update sampler with task losses
        sampler.update(avg_task_losses)
        
        # Backward pass and optimization
        loss = losses.mean()
        loss.backward()
        optimizer.step()

    # Create visualization of per-task losses
    log_steps = ex.info['log_steps']
    losses_subtasks = ex.info['losses_subtasks']
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
    gif_path = 'per_task_loss_animation_odm.gif'
    ani.save(
        gif_path,
        writer='pillow',
        fps=200,
        progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
    )
    ex.add_artifact(gif_path)
    plt.close(fig)

    # Print final metrics
    if verbose:
        print(f"\nFinal average accuracy: {ex.info['accuracies'][-1]:.4f}")
        print(f"Animation saved to {gif_path}")

    return ex.info['accuracies'][-1]
