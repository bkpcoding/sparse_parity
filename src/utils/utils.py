import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from typing import Dict, List
import torch.nn as nn
import random
import torch

def create_loss_animation(
    losses_subtasks: Dict[str, List[float]],
    log_steps: List[int],
    optimizer_name: str,
    verbose: bool = False
) -> str:
    """
    Create an animation of per-task loss over training steps.
    
    Parameters
    ----------
    losses_subtasks : Dict[str, List[float]]
        Dictionary mapping task IDs to their loss histories
    log_steps : List[int]
        List of steps at which losses were logged
    optimizer_name : str
        Name of the optimizer (for filename)
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    str
        Path to the saved GIF file
    """
    n_tasks = len(losses_subtasks)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Per-Task Loss Over Training Steps')
    ax.set_yscale('log')  # Use log scale for better visibility

    # Generate colors for each task
    colors = cm.viridis(np.linspace(0, 1, n_tasks))

    # Initialize lines for each task
    lines = []
    for i in range(n_tasks):
        line, = ax.plot([], [], color=colors[i], alpha=0.5)
        lines.append(line)

    # Set axis limits
    all_losses = [loss for task_loss in losses_subtasks.values() for loss in task_loss]
    min_step = min(log_steps)
    max_step = max(log_steps)
    min_loss = max(1e-6, min(all_losses))  # Avoid zero for log scale
    max_loss = max(all_losses)
    ax.set_xlim(min_step, max_step)
    ax.set_ylim(min_loss * 0.9, max_loss * 1.1)

    # Animation update function
    def update(frame):
        for i in range(n_tasks):
            x_data = log_steps[:frame+1]
            y_data = losses_subtasks[str(i)][:frame+1]
            lines[i].set_data(x_data, y_data)
        return lines

    # Create animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(log_steps), 
        interval=50, 
        blit=True,
        repeat=False
    )

    # Save as GIF
    gif_path = f'per_task_loss_animation_{optimizer_name}.gif'
    ani.save(
        gif_path,
        writer='pillow', 
        fps=60,
        progress_callback=lambda i, n: print(f"Saving frame {i}/{n}") if verbose else None
    )
    plt.close(fig)
    
    if verbose:
        print(f"Animation saved to {gif_path}")
    
    return gif_path


def create_model(n_tasks, n, width, depth, activation_fn, device, dtype):
    """Create and initialize the MLP model."""
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(n_tasks + n, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 2))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    return nn.Sequential(*layers).to(device)


def create_transformer(n_tasks, n, n_heads=4, n_layers=2, d_model=128, d_ff=512, device='cuda', dtype=torch.float32):
    """Create and initialize a Transformer model for the parity task.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks
    n : int
        Input dimension (excluding task embedding)
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    d_model : int
        Dimension of the model's hidden states
    d_ff : int
        Dimension of feed-forward network
    device : str
        Device to put the model on
    dtype : torch.dtype
        Data type for the model parameters
        
    Returns
    -------
    nn.Module
        The transformer model
    """
    class ParityTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Input projection layer (from input dimension to d_model)
            self.input_proj = nn.Linear(n_tasks + n, d_model)
            
            # Transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                batch_first=True
            )
            
            # Stack of transformer layers
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
            
            # Output projection layer
            self.output_proj = nn.Linear(d_model, 2)
            
        def forward(self, x):
            # Project input to d_model dimensions
            x = self.input_proj(x)
            
            # Add dummy sequence dimension (transformer expects [batch, seq_len, features])
            x = x.unsqueeze(1)
            
            # Pass through transformer
            x = self.transformer(x)
            
            # Remove sequence dimension and project to output
            x = x.squeeze(1)
            x = self.output_proj(x)
            
            return x
    
    return ParityTransformer().to(device=device, dtype=dtype)

