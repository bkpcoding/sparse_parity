#!/usr/bin/env python
# coding: utf-8

import os
import random
from collections import defaultdict

import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.task_generator import get_random_tasks, get_one_bit_different_tasks
from src.trainer.trainer import SparseParityTrainer
from src.utils.utils import create_model, create_loss_animation

@hydra.main(config_path="../configs", config_name="experiment/transfer_experiment")
def run_transfer_experiment(cfg: DictConfig):
    """Run transfer learning experiment comparing one-bit-different vs random task orderings."""
    
    # Initialize wandb
    if cfg.experiment.use_wandb:
        wandb.init(
            project="sparse-parity",
            config=dict(cfg),
            name=f"transfer_{cfg.experiment.comparison_mode}"
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
        
        # Run one-bit difference comparison
        if cfg.experiment.comparison_mode in ['one_bit_diff', 'both']:
            task_subsets = get_one_bit_different_tasks(cfg.n, cfg.k, cfg.n_tasks)
            print(f"One-bit difference tasks: {task_subsets}")
            
            # Create model and trainer
            model = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, 
                               activation_fn, device, dtype)
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=cfg.experiment.lr, 
                                        weight_decay=cfg.experiment.weight_decay)
            
            trainer = SparseParityTrainer(
                model=model,
                optimizer=optimizer,
                config=cfg,
                task_indices=list(range(cfg.n_tasks)),
                task_subsets=task_subsets
            )

            # Train model
            steps, loss_data = trainer.train()
            ensemble_results['one_bit'].append(steps)

            # Create animation for the last ensemble
            if is_last_ensemble and loss_data and cfg.plot_losses:
                log_steps, losses_subtasks = loss_data
                create_loss_animation(losses_subtasks, log_steps, 'one_bit_diff')

        # Run random tasks comparison
        if cfg.experiment.comparison_mode in ['random', 'both']:
            task_subsets = get_random_tasks(cfg.n, cfg.k, cfg.n_tasks)
            print(f"Random tasks: {task_subsets}")
            
            # Create model and trainer
            model = create_model(cfg.n_tasks, cfg.n, cfg.width, cfg.depth, 
                               activation_fn, device, dtype)
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=cfg.experiment.lr, 
                                        weight_decay=cfg.experiment.weight_decay)
            
            trainer = SparseParityTrainer(
                model=model,
                optimizer=optimizer,
                config=cfg,
                task_indices=list(range(cfg.n_tasks)),
                task_subsets=task_subsets
            )

            # Train model
            steps, loss_data = trainer.train()
            ensemble_results['random'].append(steps)

            # Create animation for the last ensemble
            if is_last_ensemble and loss_data:
                log_steps, losses_subtasks = loss_data
                create_loss_animation(losses_subtasks, log_steps, 'random')

        # Log results for this ensemble
        for method in ensemble_results:
            if len(ensemble_results[method]) > 0:
                mean_steps = np.mean(ensemble_results[method])
                std_steps = np.std(ensemble_results[method])
                
                print(f"{method.upper()} method:")
                print(f"  Mean steps: {mean_steps:.2f}")
                print(f"  Std steps: {std_steps:.2f}")
                
                results[f'{method}_mean'] = mean_steps
                results[f'{method}_std'] = std_steps
                
                if cfg.experiment.use_wandb:
                    wandb.log({
                        f'convergence_steps_{method}_mean': mean_steps,
                        f'convergence_steps_{method}_std': std_steps
                    })

    # Log comparison metrics if running both methods
    if cfg.experiment.comparison_mode == 'both':
        one_bit_mean = results['one_bit_mean']
        random_mean = results['random_mean']
        mean_diff = abs(one_bit_mean - random_mean)
        print(f"\nMean difference (One-bit vs Random): {mean_diff:.2f} steps")
        results['mean_difference'] = mean_diff
        
        if cfg.experiment.use_wandb:
            wandb.log({
                'convergence_mean_difference': mean_diff,
                'ensemble_results': dict(ensemble_results)
            })

    # Close wandb
    if cfg.experiment.use_wandb:
        wandb.finish()

    return results

if __name__ == "__main__":
    run_transfer_experiment()
