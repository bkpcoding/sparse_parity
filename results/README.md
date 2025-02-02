# Results for some of the experiments

## Transfer Learning
Here we compare the performance of ensemble of networks trained on two tasks. In the first case, the tasks are random parity tasks, i.e., the input bits of the tasks are random for each task, whereas in the second case, the tasks have some shared bits in common. Hypothesis: If the network converges faster in the second case, then there is transfer learning happening and we have found a tasks that represents the common scenarios of shared features in the input space.

### Results: 
Ensemble (over 5 runs) Statistics:

Case 1:
  Mean steps: 44900.00
  Std steps: 19120.41

Case 2:
  Mean steps: 19450.00
  Std steps: 731.44

Found the task that represents the common features in input space akin to more real-world tasks.

## Model Size
In this experiment, we compare the performance of different model sizes on the same task (Case 2) interms of convergence rate.

### Results:
- Depth 2: 21000 steps
- Depth 4: 9000 steps
- Depth 6: 8000 steps
- Depth 8: 32250 steps
- Depth 10: 30500 steps

Different model sizes converge at different rates, and the relationship is not monotonic. The implications of this are that using DoReMi might give different results for different model sizes.

## 