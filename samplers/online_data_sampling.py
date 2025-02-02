import numpy as np
import torch
from typing import List, Dict, Tuple
import math
from collections import defaultdict
from .base_sampler import BaseSampler

class ODMSampler(BaseSampler):
    """
    Implementation of Online Data Mixing for sparse parity tasks.
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
        self._tasks = []
        base_bits = sorted(list(np.random.choice(range(n), k, replace=False)))
        
        # First task uses k-1 bits
        self._tasks.append(tuple(base_bits[:k-1]))
        
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
            
            self._tasks.append(tuple(sorted(new_bits)))

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
                y = torch.sum(bits[:, list(self._tasks[task_idx])], dim=1) % 2
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
            y = torch.sum(bits[:, list(self._tasks[task_idx])], dim=1) % 2
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

    @property
    def tasks(self) -> List[tuple]:
        """Get the list of tasks (bit combinations)."""
        return self._tasks

