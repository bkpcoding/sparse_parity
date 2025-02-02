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
from torch import Tensor

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Optional
import torch
from omegaconf import DictConfig
from utils.utils import * 
from utils.fast_dataloader import FastTensorDataLoader
import wandb  # Add this import at the top with other imports
from muon import Muon  # Add this import at the top
import os
import torch.distributed as dist  # Add this import at the top

@dataclass
class OptimizerConfig:
    optimizer_mode: str  # 'adamw', 'sgd', 'muon', or 'both'
    sgd_lr: Optional[float] = None
    momentum: Optional[float] = None
    adamw_lr: Optional[float] = None
    weight_decay: Optional[float] = None
    muon_lr: Optional[float] = None
    muon_momentum: Optional[float] = 0.95
    muon_nesterov: Optional[bool] = True
    muon_ns_steps: Optional[int] = 5

@dataclass
class Config:
    n_tasks: int
    n: int
    k: int
    alpha: float
    offset: int
    D: int
    width: int
    depth: int
    activation: str
    steps: int
    batch_size: int
    test_points: int
    test_points_per_task: int
    stop_early: bool
    device: str
    dtype: str
    log_freq: int
    verbose: bool
    n_ensembles: int
    use_wandb: bool
    optimizer: OptimizerConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def train_model(model, Ss, n_tasks, n, k, alpha, offset, D, batch_size, optimizer_type,
                optimizer_params, test_points, test_points_per_task, steps, device, 
                dtype, log_freq, stop_early=False, verbose=False, store_losses=False):
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
    
    # Create optimizer based on type
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=optimizer_params['lr'],
                                    weight_decay=optimizer_params['weight_decay'])
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                  lr=optimizer_params['lr'],
                                  momentum=optimizer_params['momentum'])
    elif optimizer_type == 'muon':
        # Filter parameters based on dimensionality
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        
        optimizers = [
            Muon(muon_params, 
                 lr=optimizer_params['lr'],
                 momentum=optimizer_params['momentum'],
                 nesterov=optimizer_params['nesterov'],
                 ns_steps=optimizer_params['ns_steps']),
            torch.optim.AdamW(other_params, lr=optimizer_params['lr']*0.015)  # scaled down lr for other params
        ]
        optimizer = optimizers  # we'll handle this case specially in the training loop
    
    losses = []
    early_stop_triggers = []
    
    # Initialize loss tracking if needed
    if store_losses:
        log_steps = []
        losses_subtasks = {str(i): [] for i in range(n_tasks)}
    
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
                    return step, (log_steps, losses_subtasks) if store_losses else None
                
                # Store per-task losses if requested
                if store_losses:
                    log_steps.append(step)
                    for i in range(n_tasks):
                        x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i],
                                           sizes=[test_points_per_task], device=device, dtype=dtype)
                        y_i_pred = model(x_i)
                        losses_subtasks[str(i)].append(loss_fn(y_i_pred, y_i).item())
                
                # Early stopping logic
                if stop_early:
                    if step > 4000 and len(losses) >= 2 and losses[-1] > losses[-2]:
                        early_stop_triggers.append(True)
                    else:
                        early_stop_triggers.append(False)
                    if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                        return step
                    early_stop_triggers = early_stop_triggers[-10:]
        
        optimizer.zero_grad() if not isinstance(optimizer, list) else [opt.zero_grad() for opt in optimizer]
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
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.step()
        else:
            optimizer.step()
    
    return steps, (log_steps, losses_subtasks) if store_losses else None

@hydra.main(config_path="configs", config_name="config")
def run(cfg: DictConfig):
    # Set environment variables for distributed processing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize distributed processing for Muon
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )
    
    # Initialize wandb only if use_wandb is True
    if cfg.use_wandb:
        wandb.init(
            project="sparse-parity",
            config=dict(cfg),
            name=f"run_{cfg.optimizer.optimizer_mode}"
        )

    # Initialize results dictionary
    results = {}

    # Convert string dtype to torch.dtype
    if cfg.dtype == "float32":
        dtype = torch.float32
    elif cfg.dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")

    # Convert device string to torch.device
    device = torch.device(cfg.device)
    
    # Set random seeds
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)  # You can add seed to config if needed
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    
    if cfg.activation == 'ReLU':
        activation_fn = nn.ReLU
    elif cfg.activation == 'Tanh':
        activation_fn = nn.Tanh
    elif cfg.activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f"Unrecognized activation function: {cfg.activation}")

    # Initialize lists to store results for each ensemble
    ensemble_results = defaultdict(list)
    
    for ensemble_idx in range(cfg.n_ensembles):
        print(f"\nRunning ensemble {ensemble_idx + 1}/{cfg.n_ensembles}")
        is_last_ensemble = ensemble_idx == cfg.n_ensembles - 1
        
        # Generate random tasks (same tasks for both optimizers)
        Ss_random = get_random_tasks(cfg.n, cfg.k, cfg.n_tasks)
        print(f"Random tasks: {Ss_random}")
        
        # Run AdamW comparison
        if cfg.optimizer.optimizer_mode in ['adamw', 'both']:
            mlp_adamw = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, activation_fn, device, dtype)
            adamw_params = {'lr': cfg.optimizer.adamw_lr, 'weight_decay': cfg.optimizer.weight_decay}
            
            steps_adamw, loss_data = train_model(
                mlp_adamw, Ss_random, cfg.n_tasks, cfg.n, cfg.k, cfg.alpha, cfg.offset, cfg.D,
                cfg.batch_size, 'adamw', adamw_params, cfg.test_points, cfg.test_points_per_task,
                cfg.steps, device, dtype, cfg.log_freq, cfg.stop_early, cfg.verbose,
                store_losses=is_last_ensemble
            )
            ensemble_results['adamw'].append(steps_adamw)
            
            if is_last_ensemble and loss_data:
                log_steps, losses_subtasks = loss_data
                gif_path = create_loss_animation(
                    losses_subtasks,
                    log_steps,
                    'adamw'
                )
        
        # Run SGD comparison
        if cfg.optimizer.optimizer_mode in ['sgd', 'both']:
            mlp_sgd = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, activation_fn, device, dtype)
            sgd_params = {'lr': cfg.optimizer.sgd_lr, 'momentum': cfg.optimizer.momentum}
            
            steps_sgd, loss_data = train_model(
                mlp_sgd, Ss_random, cfg.n_tasks, cfg.n, cfg.k, cfg.alpha, cfg.offset, cfg.D,
                cfg.batch_size, 'sgd', sgd_params, cfg.test_points, cfg.test_points_per_task,
                cfg.steps, device, dtype, cfg.log_freq, cfg.stop_early, cfg.verbose,
                store_losses=is_last_ensemble
            )
            ensemble_results['sgd'].append(steps_sgd)
            
            if is_last_ensemble and loss_data:
                log_steps, losses_subtasks = loss_data
                gif_path = create_loss_animation(
                    losses_subtasks,
                    log_steps,
                    'sgd'
                )
    
        # Run Muon comparison
        if cfg.optimizer.optimizer_mode in ['muon', 'both']:
            mlp_muon = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, activation_fn, device, dtype)
            muon_params = {
                'lr': cfg.optimizer.muon_lr,
                'momentum': cfg.optimizer.muon_momentum,
                'nesterov': cfg.optimizer.muon_nesterov,
                'ns_steps': cfg.optimizer.muon_ns_steps
            }
            
            steps_muon, loss_data = train_model(
                mlp_muon, Ss_random, cfg.n_tasks, cfg.n, cfg.k, cfg.alpha, cfg.offset, cfg.D,
                cfg.batch_size, 'muon', muon_params, cfg.test_points, cfg.test_points_per_task,
                cfg.steps, device, dtype, cfg.log_freq, cfg.stop_early, cfg.verbose,
                store_losses=is_last_ensemble
            )
            ensemble_results['muon'].append(steps_muon)
            
            if is_last_ensemble and loss_data:
                log_steps, losses_subtasks = loss_data
                gif_path = create_loss_animation(
                    losses_subtasks,
                    log_steps,
                    'muon'
                )
            
            muon_mean = np.mean(ensemble_results['muon'])
            muon_std = np.std(ensemble_results['muon'])
            print(f"Muon optimizer:")
            print(f"  Mean steps: {muon_mean:.2f}")
            print(f"  Std steps: {muon_std:.2f}")
            results['muon_mean'] = muon_mean
            results['muon_std'] = muon_std
            if cfg.use_wandb:
                wandb.log({
                    'convergence_steps_muon_mean': muon_mean,
                    'convergence_steps_muon_std': muon_std
                })
    
    # Calculate and print ensemble statistics
    print("\nEnsemble Statistics:")
    
    if cfg.optimizer.optimizer_mode in ['adamw', 'both']:
        adamw_mean = np.mean(ensemble_results['adamw'])
        adamw_std = np.std(ensemble_results['adamw'])
        print(f"AdamW optimizer:")
        print(f"  Mean steps: {adamw_mean:.2f}")
        print(f"  Std steps: {adamw_std:.2f}")
        results['adamw_mean'] = adamw_mean
        results['adamw_std'] = adamw_std
        if cfg.use_wandb:
            wandb.log({
                'convergence_steps_adamw_mean': adamw_mean,
                'convergence_steps_adamw_std': adamw_std
            })
    
    if cfg.optimizer.optimizer_mode in ['sgd', 'both']:
        sgd_mean = np.mean(ensemble_results['sgd'])
        sgd_std = np.std(ensemble_results['sgd'])
        print(f"SGD optimizer:")
        print(f"  Mean steps: {sgd_mean:.2f}")
        print(f"  Std steps: {sgd_std:.2f}")
        results['sgd_mean'] = sgd_mean
        results['sgd_std'] = sgd_std
        if cfg.use_wandb:
            wandb.log({
                'convergence_steps_sgd_mean': sgd_mean,
                'convergence_steps_sgd_std': sgd_std
            })
    
    if cfg.optimizer.optimizer_mode == 'both':
        mean_diff = abs(adamw_mean - sgd_mean)
        print(f"\nMean difference: {mean_diff:.2f} steps")
        results['mean_difference'] = mean_diff
        if cfg.use_wandb:
            wandb.log({
                'convergence_mean_difference': mean_diff,
                'ensemble_results': dict(ensemble_results)
            })

    # Close wandb run if it was initialized
    if cfg.use_wandb:
        wandb.finish()
    
    return results

if __name__ == "__main__":
    run()
