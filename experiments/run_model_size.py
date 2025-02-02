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

from src.data.task_generator import get_one_bit_different_tasks
from src.trainer.trainer import SparseParityTrainer
from src.utils.utils import create_model

@hydra.main(config_path="../configs", config_name="experiment/model_size_experiment")
def run_model_size_experiment(cfg: DictConfig):
    """Run model size comparison experiment."""
    
    # Set environment variables for distributed processing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize distributed processing
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )

    # Initialize wandb
    if cfg.experiment.use_wandb:
        wandb.init(
            project="sparse-parity",
            config=dict(cfg),
            name="model_size_analysis"
        )

    # Set up device and dtype
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)

    # Set random seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Get activation function
    activation_map = {
        'ReLU': torch.nn.ReLU,
        'Tanh': torch.nn.Tanh,
        'Sigmoid': torch.nn.Sigmoid
    }
    activation_fn = activation_map.get(cfg.experiment.activation)
    if activation_fn is None:
        raise ValueError(f"Unrecognized activation function: {cfg.experiment.activation}")

    # Initialize results
    results = {}
    ensemble_results = defaultdict(list)

    for ensemble_idx in range(cfg.experiment.n_ensembles):
        print(f"\nRunning ensemble {ensemble_idx + 1}/{cfg.experiment.n_ensembles}")
        is_last_ensemble = ensemble_idx == cfg.experiment.n_ensembles - 1
        
        # Generate tasks with one-bit differences
        task_subsets = get_one_bit_different_tasks(
            cfg.experiment.n,
            cfg.experiment.k,
            cfg.experiment.n_tasks
        )
        task_indices = list(range(cfg.experiment.n_tasks))
        print(f"Tasks: {task_subsets}")
        # Run experiments for each depth
        for depth in cfg.experiment.depths:
            model_key = f'depth_{depth}'
            print(f"\nTraining model with depth {depth}")
            
            # Create model
            model = create_model(
                n_tasks=cfg.experiment.n_tasks,
                n=cfg.experiment.n,
                width=cfg.experiment.base_width,
                depth=depth,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
            )

            # Create optimizer based on config
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optimizer.adamw_lr,
                weight_decay=cfg.optimizer.weight_decay
            )

            # Create trainer
            trainer = SparseParityTrainer(
                model=model,
                optimizer=optimizer,
                config=cfg.experiment,  # Pass experiment config
                task_indices=task_indices,
                task_subsets=task_subsets
            )

            # Train model
            steps, loss_data = trainer.train()
            ensemble_results[model_key].append(steps)

            # Create animation for the last ensemble
            if is_last_ensemble and loss_data and cfg.experiment.plot_losses:
                log_steps, losses_subtasks = loss_data
                

            # Log results for this depth
            if len(ensemble_results[model_key]) > 0:
                mean_steps = np.mean(ensemble_results[model_key])
                std_steps = np.std(ensemble_results[model_key])
                
                print(f"Depth {depth}:")
                print(f"  Mean steps: {mean_steps:.2f}")
                print(f"  Std steps: {std_steps:.2f}")
                
                results[f'{model_key}_mean'] = mean_steps
                results[f'{model_key}_std'] = std_steps
                
                if cfg.experiment.use_wandb:
                    wandb.log({
                        f'convergence_steps_depth_{depth}_mean': mean_steps,
                        f'convergence_steps_depth_{depth}_std': std_steps
                    })

    # Print comparison summary
    print("\nModel Size Comparison:")
    for depth in cfg.experiment.depths:
        print(f"Depth {depth}: {ensemble_results[f'depth_{depth}']} steps")

    if cfg.experiment.use_wandb:
        wandb.log({
            'ensemble_results': dict(ensemble_results)
        })
        wandb.finish()

    return results

if __name__ == "__main__":
    run_model_size_experiment() 