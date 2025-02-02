from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch

class BaseSampler(ABC):
    """Base class for all samplers."""
    
    @abstractmethod
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of data."""
        pass
    
    @abstractmethod
    def update(self, losses: Dict[int, float]):
        """Update sampler based on observed losses."""
        pass
    
    @abstractmethod
    def get_task_indices(self) -> List[int]:
        """Get indices of tasks in the current batch."""
        pass
    
    @property
    @abstractmethod
    def tasks(self) -> List[tuple]:
        """Get the list of tasks (bit combinations)."""
        pass