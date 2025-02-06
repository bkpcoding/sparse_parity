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
from src.utils.utils import create_model, create_transformer

@hydra.main(config_path="../configs", config_name="experiment/architecture_experiment")
def run_architecture_experiment(cfg: DictConfig):
    """Run architecture comparison experiment between MLP and Transformer models."""
    
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
            name="architecture_analysis"
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

    # Get activation function for MLP
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

        # Run experiments for each architecture
        architectures = {
            'mlp': lambda: create_model(
                n_tasks=len(task_indices),
                n=cfg.n,
                width=cfg.experiment.mlp_width,
                depth=cfg.experiment.mlp_depth,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
            ),
            'transformer': lambda: create_transformer(
                n_tasks=len(task_indices),
                n=cfg.n,
                n_heads=cfg.experiment.transformer_heads,
                n_layers=cfg.experiment.transformer_layers,
                d_model=cfg.experiment.transformer_d_model,
                d_ff=cfg.experiment.transformer_d_ff,
                device=device,
                dtype=dtype
            )
        }

        for arch_name, model_fn in architectures.items():
            print(f"\nTraining {arch_name.upper()} model")
            
            # Create model
            model = model_fn()

            # Create optimizer based on config
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.experiment.optimizer.adamw_lr,
                weight_decay=cfg.experiment.optimizer.weight_decay
            )

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
            ensemble_results[arch_name].append(steps)

            # Create animation for the last ensemble
            if is_last_ensemble and loss_data and cfg.experiment.plot_losses:
                log_steps, losses_subtasks = loss_data

            # Log results for this architecture
            if len(ensemble_results[arch_name]) > 0:
                mean_steps = np.mean(ensemble_results[arch_name])
                std_steps = np.std(ensemble_results[arch_name])
                
                print(f"{arch_name.upper()}:")
                print(f"  Mean steps: {mean_steps:.2f}")
                print(f"  Std steps: {std_steps:.2f}")
                
                results[f'{arch_name}_mean'] = mean_steps
                results[f'{arch_name}_std'] = std_steps
                
                if cfg.experiment.use_wandb:
                    wandb.log({
                        f'convergence_steps_{arch_name}_mean': mean_steps,
                        f'convergence_steps_{arch_name}_std': std_steps
                    })

    # Print comparison summary
    print("\nArchitecture Comparison:")
    for arch_name in architectures.keys():
        print(f"{arch_name.upper()}: {ensemble_results[arch_name]} steps")

    if cfg.experiment.use_wandb:
        wandb.log({
            'ensemble_results': dict(ensemble_results)
        })
        wandb.finish()

    return results

if __name__ == "__main__":
    run_architecture_experiment()
