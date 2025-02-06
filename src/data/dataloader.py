import torch
import numpy as np
from typing import List, Tuple, Optional


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


class TemperatureDataLoader:
    """
    A DataLoader that implements temperature-based upsampling for handling imbalanced datasets.
    Calculates sampling probabilities based on dataset sizes and a temperature parameter τ.
    Higher τ values lead to more aggressive upsampling of minority classes/tasks.
    """
    def __init__(
        self,
        n_tasks: int,
        n: int,
        Ss: List[List[int]],
        sizes: List[int],
        batch_size: int = 32,
        temperature: float = 5.0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize temperature-based sampling dataloader.
        
        Args:
            n_tasks: Number of tasks
            n: Bit string length for sparse parity problem
            Ss: Subsets of [1, ..., n] to compute sparse parities on
            sizes: Number of samples for each subtask
            batch_size: Size of batches to generate
            temperature: Sampling temperature τ (default: 5.0)
            device: Device to put tensors on
            dtype: Data type for input tensors
        """
        self.n_tasks = n_tasks
        self.n = n
        self.Ss = Ss
        self.sizes = sizes
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.dtype = dtype
        
        # Calculate sampling probabilities using temperature
        self.sampling_probs = self._compute_sampling_probs()
        
        # Calculate expected number of samples per epoch
        self.samples_per_epoch = sum(sizes)
        
        # Calculate number of batches
        self.n_batches = self.samples_per_epoch // batch_size
        if self.samples_per_epoch % batch_size != 0:
            self.n_batches += 1
            
    def _compute_sampling_probs(self) -> np.ndarray:
        """
        Compute temperature-adjusted sampling probabilities.
        p(i;τ) = |Di|^(1/τ) / Σ_j |Dj|^(1/τ)
        """
        sizes_array = np.array(self.sizes)
        temp_adjusted = np.power(sizes_array, 1/self.temperature)
        return temp_adjusted / temp_adjusted.sum()
    
    def sample_batch_sizes(self) -> List[int]:
        """
        Sample batch sizes for each task based on temperature-adjusted probabilities.
        """
        task_counts = np.random.multinomial(self.batch_size, self.sampling_probs)
        return task_counts.tolist()
    
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_batch >= self.n_batches:
            raise StopIteration
            
        # Sample sizes for each task in this batch
        batch_sizes = self.sample_batch_sizes()
        
        # Generate batch using the helper function
        batch = get_batch(
            n_tasks=self.n_tasks,
            n=self.n,
            Ss=self.Ss,
            codes=list(range(len(self.Ss))),
            sizes=batch_sizes,
            device=self.device,
            dtype=self.dtype
        )
        
        self.current_batch += 1
        return batch
    
    def __len__(self) -> int:
        return self.n_batches
        
    def set_temperature(self, new_temp: float):
        """Update the sampling temperature."""
        self.temperature = new_temp
        self.sampling_probs = self._compute_sampling_probs()

class OnlineDataMixingSampler:
    """
    A DataLoader-like object that implements Online Data Mixing (ODM) based on the Exp3 algorithm.
    Uses losses as rewards to dynamically adjust sampling probabilities across datasets.
    Creates homogeneous batches (all samples from same dataset) to accurately attribute rewards.
    """
    def __init__(
        self,
        n_tasks: int,
        n: int,
        Ss: List[List[int]],
        sizes: List[int],
        batch_size: int = 32,
        alpha_decay: float = 0.9,  # Moving average decay factor
        warmup_steps: int = 1000,  # Number of steps before starting ODM
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.n_tasks = n_tasks
        self.n = n
        self.Ss = Ss
        self.sizes = sizes
        self.batch_size = batch_size
        self.alpha = alpha_decay
        self.warmup_steps = warmup_steps
        self.device = device
        self.dtype = dtype
        
        # Initialize tracking variables
        self.current_step = 0
        self.K = len(Ss)  # Number of arms/datasets
        
        # Initialize cumulative domain losses for current batch
        self.domain_losses = torch.zeros(self.K, device=device, dtype=dtype)
        
        # Initialize estimated rewards
        self.estimated_rewards = torch.zeros(self.K, device=device, dtype=dtype)
        
        # Initialize sampling probabilities uniformly
        self.sampling_probs = torch.ones(self.K, device=device, dtype=dtype) / self.K
        
        # Calculate number of batches
        min_samples = min(sizes)
        self.n_batches = min_samples // batch_size
        if min_samples % batch_size != 0:
            self.n_batches += 1
            
    def get_exploration_rate(self) -> float:
        """Calculate exploration rate for current step using decay schedule."""
        if self.current_step < self.warmup_steps:
            return 1.0 / self.K
        # Ensure exploration rate doesn't get too small
        return max(0.01, min(1.0/self.K, np.sqrt(np.log(self.K)/(self.K * max(1, self.current_step)))))
        
    def update_policy(self, dataset_idx: int, loss: float):
        """
        Update sampling distribution based on observed loss for selected dataset.
        Following Algorithm 1 from the paper exactly.
        
        Args:
            dataset_idx: Index of the dataset that was sampled
            loss: Loss value from training on the batch
        """
        if self.current_step < self.warmup_steps:
            return
            
        # Clip loss to prevent extreme values
        loss = min(max(loss, 1e-8), 1e3)
        
        # Get current probability for importance weighting
        current_prob = max(self.sampling_probs[dataset_idx].item(), 1e-8)
        
        # Calculate importance weighted reward as per paper
        importance_weighted_reward = loss / current_prob
        
        # Update moving average of rewards
        old_reward = self.estimated_rewards[dataset_idx].item()
        new_reward = self.alpha * old_reward + (1 - self.alpha) * importance_weighted_reward
        # print(f"old reward: {old_reward}, importance weighted reward: {importance_weighted_reward}, alpha: {self.alpha}, intermediate reward: {self.alpha * old_reward}, {(1 - self.alpha) * importance_weighted_reward}")
        # print(f"new reward: {new_reward}")
        self.estimated_rewards[dataset_idx] = new_reward
        
        # Calculate exploration rate
        exploration_rate = self.get_exploration_rate()
        
        # Scale rewards to prevent numerical instability
        scaled_rewards = self.estimated_rewards - self.estimated_rewards.mean()
        if torch.std(scaled_rewards) > 0:
            scaled_rewards = scaled_rewards / torch.std(scaled_rewards)
            
        # Compute probabilities using stable softmax
        logits = scaled_rewards / exploration_rate
        max_logit = torch.max(logits)
        exp_logits = torch.exp(logits - max_logit)
        softmax_probs = exp_logits / (exp_logits.sum() + 1e-8)
        
        # Ensure minimum exploration probability for each arm
        min_prob = exploration_rate / self.K
        self.sampling_probs = (1 - self.K * min_prob) * softmax_probs + min_prob
        
        # Final numerical stability checks
        self.sampling_probs = torch.clamp(self.sampling_probs, min=1e-8)
        self.sampling_probs = self.sampling_probs / self.sampling_probs.sum()

    def sample_dataset(self) -> int:
        """Sample a dataset index according to current policy."""
        return torch.multinomial(self.sampling_probs, num_samples=1).item()
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.current_batch >= self.n_batches:
            raise StopIteration
            
        # Sample a single dataset according to policy
        dataset_idx = self.sample_dataset()
        
        # Generate homogeneous batch from selected dataset
        batch = get_batch(
            n_tasks=self.n_tasks,
            n=self.n, 
            Ss=[self.Ss[dataset_idx]],  # Only use selected dataset
            codes=[dataset_idx],
            sizes=[self.batch_size],  # Full batch size from selected dataset
            device=self.device,
            dtype=self.dtype
        )
        
        self.current_batch += 1
        self.current_step += 1
        
        return batch[0], batch[1], dataset_idx
        
    def __len__(self) -> int:
        return self.n_batches

# Example usage
if __name__ == "__main__":
    # Example parameters
    n_tasks = 3
    n = 10
    Ss = [[1,3,5], [2,4,6], [1,2,3]]
    sizes = [1000, 100, 50]  # Imbalanced dataset sizes
    
    # Create dataloader with temperature τ = 5
    loader = TemperatureDataLoader(
        n_tasks=n_tasks,
        n=n,
        Ss=Ss,
        sizes=sizes,
        batch_size=32,
        temperature=5.0
    )
    
    # Example of getting a batch
    x, y = next(iter(loader))
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
    
    # Example of changing temperature during training (as in COOLDOWN)
    loader.set_temperature(1.0)  # Reduce temperature later in training