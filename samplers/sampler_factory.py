from typing import Dict, Any
from .online_data_sampling import ODMSampler
# Import other samplers as they are created

def create_sampler(sampler_type: str, config: Dict[str, Any]):
    """
    Factory function to create samplers based on config.
    
    Args:
        sampler_type: Type of sampler to create ('odm', 'uniform', etc.)
        config: Configuration dictionary with sampler parameters
    """
    if sampler_type.lower() == 'odm':
        return ODMSampler(
            n=config['n'],
            k=config['k'],
            n_tasks=config['n_tasks'],
            D=config['D'],
            alpha=config.get('alpha', 1.5),
            offset=config.get('offset', 0),
            batch_size=config['batch_size'],
            device=config['device'],
            warmup_steps=config.get('warmup_steps', 100),
            ma_alpha=config.get('ma_alpha', 0.9)
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}") 