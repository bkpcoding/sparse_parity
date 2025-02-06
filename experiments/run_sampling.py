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

from src.data.task_generator import get_random_tasks
from src.trainer.trainer import SparseParityTrainer
from src.utils.utils import create_model, create_loss_animation

@hydra.main(config_path="../configs", config_name="experiment/sampling_experiment")
def run_sampling_experiment(cfg: DictConfig):
    """Run experiment comparing uniform vs temperature-based sampling."""
    
    # Initialize wandb
    if cfg.experiment.use_wandb:
        wandb.init(
            project="sparse-parity",
            config=dict(cfg),
            name=f"sampling_{cfg.experiment.comparison_mode}"
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
        
        # Generate random tasks for this experiment
        task_subsets = get_random_tasks(cfg.n, cfg.k, cfg.n_tasks)
        
        # Run each sampling method based on comparison_mode
        sampling_methods = []
        if cfg.experiment.comparison_mode == 'all':
            sampling_methods = ['uniform', 'temperature', 'online_mixing']
        elif cfg.experiment.comparison_mode == 'both':
            sampling_methods = ['uniform', 'temperature']
        else:
            sampling_methods = [cfg.experiment.comparison_mode]

        for method in sampling_methods:
            print(f"Running with {method} sampling...")
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
                task_subsets=task_subsets,
                sampler_type=method
            )

            steps, loss_data = trainer.train()
            ensemble_results[method].append(steps)
            if is_last_ensemble and loss_data and cfg.experiment.plot_losses:
                log_steps, losses_subtasks = loss_data
                create_loss_animation(losses_subtasks, log_steps, method)

        # Log results for this ensemble
        for method in ensemble_results:
            if len(ensemble_results[method]) > 0:
                mean_steps = np.mean(ensemble_results[method])
                std_steps = np.std(ensemble_results[method])
                
                print(f"{method.upper()} sampling:")
                print(f"  Mean steps: {mean_steps:.2f}")
                print(f"  Std steps: {std_steps:.2f}")
                
                results[f'{method}_mean'] = mean_steps
                results[f'{method}_std'] = std_steps
                
                if cfg.experiment.use_wandb:
                    wandb.log({
                        f'convergence_steps_{method}_mean': mean_steps,
                        f'convergence_steps_{method}_std': std_steps
                    })

    # Log comparison metrics if running multiple methods
    if len(sampling_methods) > 1:
        # Print pairwise comparisons
        methods = list(results.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i].replace('_mean', '')
                method2 = methods[j].replace('_mean', '')
                mean1 = results[f'{method1}_mean']
                mean2 = results[f'{method2}_mean']
                mean_diff = abs(mean1 - mean2)
                print(f"\nMean difference ({method1} vs {method2}): {mean_diff:.2f} steps")
                results[f'mean_difference_{method1}_{method2}'] = mean_diff
                
                if cfg.experiment.use_wandb:
                    wandb.log({
                        f'convergence_mean_difference_{method1}_{method2}': mean_diff
                    })

    # Close wandb
    if cfg.experiment.use_wandb:
        wandb.finish()

    return results

if __name__ == "__main__":
    run_sampling_experiment()