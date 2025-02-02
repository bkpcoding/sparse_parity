# SPARSE PARITY 

Sparse parity is a computational problem that is known to be hard. The utility of this problem is in studying the learning of neural networks with quick feedback loops. This is very much inspired by [Hidden Progress in Deep Learning:
SGD Learns Parities Near the Computational Limit](https://proceedings.neurips.cc/paper_files/paper/2022/file/884baf65392170763b27c914087bde01-Paper-Conference.pdf), you can refer to the paper for more details on the hardness of the problem and the optimality of SGD in solving it. The codebase is built on top of [this repo](https://github.com/ejmichaud/quantization-model).

Some of the experiments (or hypotheses) that you can do using sparse parity are:
- How does the performance of SGD compare with other optimizers?
- How does varying the depth and width of the network affect the convergence rate?
- Can neural networks learn common features in the sparse parity tasks to converge faster?
- Can we identify the common circuits responsible for tasks sharing the same parity?
- How does sampling from a finite dataset compare to sampling from the infinite dataset?

