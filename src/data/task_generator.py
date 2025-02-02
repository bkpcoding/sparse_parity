import random
import torch

def get_one_bit_different_tasks(n, k, n_tasks):
    """Generate n_tasks where each consecutive task differs by exactly one position.
    
    Parameters
    ----------
    n : int
        Length of bit string
    k : int
        Number of active bits in each task
    n_tasks : int
        Number of tasks to generate
    
    Returns
    -------
    list of tuples
        List of n_tasks tasks where each consecutive pair differs by one bit
    """
    tasks = []
    
    # Generate first task's bits
    current_S = sorted(random.sample(range(n), k))
    tasks.append(tuple(current_S))
    
    # Generate remaining tasks
    for _ in range(n_tasks - 1):
        next_S = current_S.copy()
        available_positions = list(set(range(n)) - set(current_S))
        
        # If no available positions, reshuffle completely
        if not available_positions:
            next_S = sorted(random.sample(range(n), k))
        else:
            bit_to_remove = random.choice(current_S)
            new_bit = random.choice(available_positions)
            next_S.remove(bit_to_remove)
            next_S.append(new_bit)
            next_S.sort()
        
        tasks.append(tuple(next_S))
        current_S = next_S
    
    return tasks

def get_random_tasks(n, k, n_tasks):
    """Generate n_tasks completely random tasks.
    
    Parameters
    ----------
    n : int
        Length of bit string
    k : int
        Number of active bits in each task
    n_tasks : int
        Number of tasks to generate
    
    Returns
    -------
    list of tuples
        List of n_tasks random tasks
    """
    tasks = []
    while len(tasks) < n_tasks:
        S = tuple(sorted(random.sample(range(n), k)))
        if S not in tasks:  # Ensure no duplicate tasks
            tasks.append(S)
    return tasks



def tasks_equal(tasks1, tasks2):
    """Compare two lists of tasks for equality."""
    if len(tasks1) != len(tasks2):
        return False
    return all(tuple(t1) == tuple(t2) for t1, t2 in zip(tasks1, tasks2))

