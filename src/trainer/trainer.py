from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from tqdm.auto import tqdm
import wandb
from omegaconf import DictConfig

from src.data.dataloader import FastTensorDataLoader, get_batch, cycle
from src.utils.utils import create_loss_animation

class SparseParityTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[Optimizer, List[Optimizer]],
        config: DictConfig,
        task_indices: List[int],
        task_subsets: List[List[int]],
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            optimizer: Single optimizer or list of optimizers
            config: Configuration object
            task_indices: List of task indices
            task_subsets: List of subsets for each task
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.task_indices = task_indices
        self.task_subsets = task_subsets
        
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize probability distribution for tasks
        self.probs = np.array([
            np.power(config.n, -config.alpha) 
            for n in range(1 + config.offset, config.n_tasks + config.offset + 1)
        ])
        self.probs = self.probs / np.sum(self.probs)
        self.cdf = np.cumsum(self.probs)
        
        # Calculate test batch sizes
        self.test_batch_sizes = [
            int(prob * config.test_points) for prob in self.probs
        ]
        
        # Initialize data loader if using fixed dataset
        if config.D != -1:
            self.train_iter = self._initialize_dataloader()
    
    def _initialize_dataloader(self) -> iter:
        """Initialize the data loader for fixed dataset size."""
        samples = np.searchsorted(self.cdf, np.random.rand(self.config.D,))
        hist, _ = np.histogram(samples, bins=self.config.n_tasks, 
                             range=(0, self.config.n_tasks-1))
        
        train_x, train_y = get_batch(
            n_tasks=self.config.n_tasks,
            n=self.config.n,
            Ss=self.task_subsets,
            codes=self.task_indices,
            sizes=hist,
            device='cpu',
            dtype=self.dtype
        )
        
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        train_loader = FastTensorDataLoader(
            train_x, train_y,
            batch_size=min(self.config.D, self.config.batch_size),
            shuffle=True
        )
        
        return cycle(train_loader)
    
    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data, either from dataloader or generated on-the-fly."""
        if self.config.D != -1:
            return next(self.train_iter)
        
        samples = np.searchsorted(self.cdf, np.random.rand(self.config.batch_size,))
        hist, _ = np.histogram(samples, bins=self.config.n_tasks, 
                             range=(0, self.config.n_tasks-1))
        
        return get_batch(
            n_tasks=self.config.n_tasks,
            n=self.config.n,
            Ss=self.task_subsets,
            codes=self.task_indices,
            sizes=hist,
            device=self.device,
            dtype=self.dtype
        )
    
    def _optimizer_step(self, loss: torch.Tensor):
        """Perform optimization step."""
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizer:
                opt.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self) -> Tuple[int, Optional[Tuple]]:
        """
        Train the model.
        
        Returns:
            steps_to_convergence: Number of steps taken
            loss_data: Tuple of (log_steps, losses_subtasks) if store_losses=True
        """
        losses = []
        early_stop_triggers = []
        
        # Initialize loss tracking if needed
        store_losses = (self.config.n_ensembles == 1)  # Store only for single runs
        if store_losses:
            log_steps = []
            losses_subtasks = {str(i): [] for i in range(self.config.n_tasks)}
        
        for step in tqdm(range(self.config.steps)):
            # Logging and evaluation
            if step % self.config.log_freq == 0:
                with torch.no_grad():
                    loss = self._evaluate_step()
                    losses.append(loss)
                    
                    # Check for convergence
                    if loss <= 1e-6:
                        if self.config.verbose:
                            print(f"Converged at step {step} with loss {loss}")
                        return step, (log_steps, losses_subtasks) if store_losses else None
                    
                    # Store per-task losses if requested
                    if store_losses:
                        self._store_task_losses(step, log_steps, losses_subtasks)
                    
                    # Early stopping check
                    if self._should_stop_early(losses, early_stop_triggers):
                        return step, (log_steps, losses_subtasks) if store_losses else None
            
            # Training step
            x, y_target = self._get_batch()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y_target)
            self._optimizer_step(loss)
            
            # Log to wandb if enabled
            if self.config.use_wandb and step % self.config.log_freq == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "step": step
                })
        
        return self.config.steps, (log_steps, losses_subtasks) if store_losses else None
    
    def _evaluate_step(self) -> float:
        """Perform evaluation step."""
        x_i, y_i = get_batch(
            n_tasks=self.config.n_tasks,
            n=self.config.n,
            Ss=self.task_subsets,
            codes=self.task_indices,
            sizes=self.test_batch_sizes,
            device=self.device,
            dtype=self.dtype
        )
        y_i_pred = self.model(x_i)
        return self.loss_fn(y_i_pred, y_i).item()
    
    def _store_task_losses(self, step: int, log_steps: List[int], 
                          losses_subtasks: Dict[str, List[float]]):
        """Store per-task losses."""
        log_steps.append(step)
        for i in range(self.config.n_tasks):
            x_i, y_i = get_batch(
                n_tasks=self.config.n_tasks,
                n=self.config.n,
                Ss=[self.task_subsets[i]],
                codes=[i],
                sizes=[self.config.test_points_per_task],
                device=self.device,
                dtype=self.dtype
            )
            y_i_pred = self.model(x_i)
            losses_subtasks[str(i)].append(self.loss_fn(y_i_pred, y_i).item())
    
    def _should_stop_early(self, losses: List[float], 
                          early_stop_triggers: List[bool]) -> bool:
        """Check if training should stop early."""
        if not self.config.stop_early:
            return False
            
        if len(losses) >= 2 and losses[-1] > losses[-2]:
            early_stop_triggers.append(True)
        else:
            early_stop_triggers.append(False)
            
        early_stop_triggers = early_stop_triggers[-10:]  # Keep last 10 only
        
        return (len(early_stop_triggers) > 10 and 
                all(early_stop_triggers[-10:]))
