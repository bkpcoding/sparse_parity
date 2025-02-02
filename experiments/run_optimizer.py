#!/usr/bin/env python
# coding: utf-8

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import wandb
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.task_generator import get_random_tasks
from src.trainer.trainer import SparseParityTrainer
from src.utils.utils import *
from muon import Muon

@hydra.main(config_path="../configs", config_name="experiment/optimizer_experiment")
def run_optimizer_experiment(cfg: DictConfig):
    """Run optimizer comparison experiment."""
    
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
    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project="sparse-parity",
            config=dict(cfg),
            name=f"run_{cfg.optimizer.optimizer_mode}"
        )

    # Set up device and dtype
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)

    # Set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)

    # Get activation function
    activation_map = {
        'ReLU': torch.nn.ReLU,
        'Tanh': torch.nn.Tanh,
        'Sigmoid': torch.nn.Sigmoid
    }
    activation_fn = activation_map.get(cfg.activation)
    if activation_fn is None:
        raise ValueError(f"Unrecognized activation function: {cfg.activation}")

    # Initialize results
    results = {}
    ensemble_results = defaultdict(list)

    for ensemble_idx in range(cfg.n_ensembles):
        print(f"\nRunning ensemble {ensemble_idx + 1}/{cfg.n_ensembles}")
        is_last_ensemble = ensemble_idx == cfg.n_ensembles - 1
        
        # Generate random tasks
        task_subsets = get_random_tasks(cfg.n, cfg.k, cfg.n_tasks)
        task_indices = list(range(cfg.n_tasks))
        print(f"Random tasks: {task_subsets}")

        # Run experiments for each optimizer type
        for opt_type in get_optimizer_types(cfg.optimizer.optimizer_mode):
            # Create model and optimizer
            model = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, 
                               activation_fn, device, dtype)
            optimizer = create_optimizer(model, opt_type, cfg.optimizer)

            # Create trainer
            trainer = SparseParityTrainer(
                model=model,
                optimizer=optimizer,
                config=cfg,
                task_indices=task_indices,
                task_subsets=task_subsets
            )

            # Train model
            steps, loss_data = trainer.train()
            ensemble_results[opt_type].append(steps)

            # Create animation for the last ensemble
            if is_last_ensemble and loss_data:
                log_steps, losses_subtasks = loss_data
                create_loss_animation(losses_subtasks, log_steps, opt_type)

            # Log results for this optimizer
            if len(ensemble_results[opt_type]) > 0:
                mean_steps = np.mean(ensemble_results[opt_type])
                std_steps = np.std(ensemble_results[opt_type])
                
                print(f"{opt_type.upper()} optimizer:")
                print(f"  Mean steps: {mean_steps:.2f}")
                print(f"  Std steps: {std_steps:.2f}")
                
                results[f'{opt_type}_mean'] = mean_steps
                results[f'{opt_type}_std'] = std_steps
                
                if cfg.use_wandb:
                    wandb.log({
                        f'convergence_steps_{opt_type}_mean': mean_steps,
                        f'convergence_steps_{opt_type}_std': std_steps
                    })

    # Log comparison metrics if running multiple optimizers
    if cfg.optimizer.optimizer_mode == 'both':
        adamw_mean = results['adamw_mean']
        sgd_mean = results['sgd_mean']
        mean_diff = abs(adamw_mean - sgd_mean)
        print(f"\nMean difference (AdamW vs SGD): {mean_diff:.2f} steps")
        results['mean_difference'] = mean_diff
        
        if cfg.use_wandb:
            wandb.log({
                'convergence_mean_difference': mean_diff,
                'ensemble_results': dict(ensemble_results)
            })

    # Close wandb
    if cfg.use_wandb:
        wandb.finish()

    return results

def get_optimizer_types(optimizer_mode: str) -> list:
    """Get list of optimizer types to run based on mode."""
    if optimizer_mode == 'both':
        return ['adamw', 'sgd', 'muon']
    return [optimizer_mode]

def create_optimizer(model, optimizer_type: str, optimizer_config):
    """Create optimizer based on type and config."""
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.adamw_lr,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config.sgd_lr,
            momentum=optimizer_config.momentum
        )
    elif optimizer_type == 'muon':
        # Filter parameters based on dimensionality
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        
        return [
            Muon(
                muon_params,
                lr=optimizer_config.muon_lr,
                momentum=optimizer_config.muon_momentum,
                nesterov=optimizer_config.muon_nesterov,
                ns_steps=optimizer_config.muon_ns_steps
            ),
            torch.optim.AdamW(
                other_params,
                lr=optimizer_config.muon_lr * 0.015
            )
        ]
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

if __name__ == "__main__":
    run_optimizer_experiment()
