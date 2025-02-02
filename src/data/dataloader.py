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
        
        import ipdb; ipdb.set_trace()
        self.current_batch += 1
        return batch
    
    def __len__(self) -> int:
        return self.n_batches
        
    def set_temperature(self, new_temp: float):
        """Update the sampling temperature."""
        self.temperature = new_temp
        self.sampling_probs = self._compute_sampling_probs()
        print(f"Updated sampling probabilities: {self.sampling_probs}")

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